"""Replay frozen room weights separately on each physical source and noise.

Diagnostic truth is never used to redesign the frozen mixture-trained weights.
"""

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.evaluation.metrics import bandpass_for_evaluation
from acoustic_array.evaluation.room_components import target_component_summary
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room


def verified_manifest(run: Path) -> dict:
    manifest = load_json(run / "manifest.json")
    for name, digest in manifest["outputs_sha256"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or sha256_file(path) != digest:
            raise ValueError("Archive hash mismatch or escaped path")
    return manifest


def parent_methods(parent: dict, prior: dict) -> set[str]:
    """Require one result per configured case/method, including historical runs."""
    methods = {"das", "fixed_constraint"} | {
        f"mpdr_load{value}" for value in parent["config"]["diagonal_loadings"]
    }
    pairs = [(row["case"], row["method"]) for row in prior["rows"]]
    cases = {case for case, _ in pairs}
    expected = prior["expected_cases"]
    if (
        expected <= 0
        or prior["completed_cases"] != expected
        or len(cases) != expected
        or len(pairs) != len(set(pairs))
        or set(pairs) != {(case, method) for case in cases for method in methods}
        or ("selected_cases" in parent and cases != set(parent["selected_cases"]))
    ):
        raise ValueError("Incomplete or duplicated MPDR parent matrix")
    return methods


def frozen_weight_path(mpdr: Path, room: Path, parent: dict, row: dict, method: str):
    """Never substitute oracle baselines for a geometry-mismatched run."""
    case = row["case"]
    if method.startswith("mpdr_load"):
        return mpdr / f"{case}_{method}_weights.npz", "weights"
    error = parent["config"].get("steering_error", {})
    if any(error.get(key, 0) != 0 for key in ("angle_offset_deg", "distance_offset_m")):
        path = mpdr / f"{case}_baseline_weights.npz"
        if not path.is_file():
            raise ValueError("Missing geometry-mismatched baseline weights")
    else:
        d, gap = row["distance_m"], row["adjacent_angle_deg"]
        size = parent["config"]["fft_size"]
        path = room / f"r{d}_gap{gap}_{size}_weights.npz"
    return path, "das" if method == "das" else "constrained"


def run_component_replay(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    if (
        set(cfg) != {"schema_version", "mpdr_run", "room_run", "cases", "methods"}
        or cfg["schema_version"] != "0.1"
    ):
        raise ValueError("Invalid replay config")
    mpdr = (config_path.parent / cfg["mpdr_run"]).resolve()
    room = (config_path.parent / cfg["room_run"]).resolve()
    verified_manifest(mpdr)
    verified_manifest(room)
    parent = load_json(mpdr / "config.expanded.json")
    if parent["parent_manifest_sha256"] != sha256_file(room / "manifest.json"):
        raise ValueError("Room does not match trained-weight parent")
    prior = load_json(mpdr / "metrics.json")
    available_methods = parent_methods(parent, prior)
    cases = cfg["cases"]
    methods = cfg["methods"]
    available = {r["case"] for r in prior["rows"]}
    if (
        not isinstance(cases, list)
        or not cases
        or any(not isinstance(case, str) for case in cases)
        or len(set(cases)) != len(cases)
        or not set(cases) <= available
    ):
        raise ValueError("Invalid case selection")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) for method in methods)
        or len(set(methods)) != len(methods)
        or not set(methods) <= available_methods
    ):
        raise ValueError("Unsupported frozen methods")
    original = parent["parent"]["config"]
    array = ArrayConfig.from_mapping(parent["parent"]["array"])
    stft = STFTConfig(parent["config"]["fft_size"], parent["config"]["fft_size"] // 2)
    calibration = parent["calibration_interval_samples"][1]
    start, stop = parent["evaluation_interval_samples"]
    with np.load(room / "sources.npz") as z:
        sources = z["sources"].copy()
    diagnosis = {d["case"]: d for d in json.loads((room / "room_diagnostics.json").read_text())}
    run = output_root.resolve() / (
        "g1_components_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    run.mkdir(parents=True, exist_ok=False)
    expanded = {
        "config": cfg,
        "parent": parent,
        "mpdr_manifest_sha256": sha256_file(mpdr / "manifest.json"),
        "room_manifest_sha256": sha256_file(room / "manifest.json"),
        "policy": "frozen-weight replay; no refitting; same filtered activity interval",
    }
    write_json(run / "config.expanded.json", expanded)
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for p in package.rglob("*.py"):
        name = p.relative_to(package).as_posix()
        dst = run / "source_snapshot" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        hashes[name] = sha256_file(dst)
    rows = []
    for index, case in enumerate(cases):
        template = next(r for r in prior["rows"] if r["case"] == case)
        with np.load(room / f"{case}_reconstruction.npz") as z:
            rirs = RoomRIR(
                z["rirs"],
                z["direct_rirs"],
                z["sources_m"],
                z["microphones_m"],
                z["direct_distances_m"],
                template["target_rt60_s"],
                diagnosis[case]["energy_absorption"],
            )
            base, direct = propagate_room(sources, array, rirs)
            built = normalize_active_scene(
                base,
                array,
                activity_components=direct,
                active_rms=original["active_rms"],
                noise_snr_db=original["noise_snr_db"],
                seed=original["seed"],
                trim_samples=original["trim_samples"],
            )
            if not np.array_equal(built.source_gains, z["gains"]) or not np.array_equal(
                built.masks, z["masks"]
            ):
                raise ValueError("Scene reconstruction mismatch")
        scene = built.scene
        direct *= built.source_gains[:, None, None]
        ref = array.reference_microphone_index
        mask = built.masks.copy()
        mask[:, :start] = False
        mask[:, stop:] = False
        for method in methods:
            path, key = frozen_weight_path(mpdr, room, parent, template, method)
            with np.load(path) as z:
                weights = z[key].copy()

            def process(x, weights=weights):
                tail = beamform_with_weights(
                    x[calibration:], array, stft, weights, block_size=original["block_size"]
                )
                return np.pad(tail, ((calibration, 0), (0, 0)))

            outputs = np.stack([process(x) for x in scene.components])
            direct_outputs = np.stack([process(x) for x in direct])
            noise = process(scene.noise)
            mixture = process(scene.mixture)
            error = float(np.max(np.abs(mixture - outputs.sum(axis=0) - noise)))
            if error > 1e-10:
                raise ValueError("Component replay is not linear")
            row = {"case": case, "method": method, "linearity_max_abs_error": error, "targets": []}
            for band in ["fullband", "evaluation_band"]:

                def filtered(x, band=band):
                    return (
                        bandpass_for_evaluation(x, array.sample_rate_hz, (500, 5000))
                        if band == "evaluation_band"
                        else x
                    )

                out = np.stack([filtered(x) for x in outputs])
                dout = np.stack([filtered(x) for x in direct_outputs])
                nout = filtered(noise)
                source_refs = filtered(scene.components[:, :, ref].T)
                direct_refs = filtered(direct[:, :, ref].T)
                mixref = filtered(scene.mixture[:, ref])
                noiseref = filtered(scene.noise[:, ref])
                for k in range(3):
                    m = mask[k]
                    input_summary = target_component_summary(
                        direct_refs[m, k],
                        direct_refs[m, k],
                        source_refs[m, k],
                        source_refs[m].sum(axis=1) - source_refs[m, k],
                        noiseref[m],
                        mixref[m],
                    )
                    output_summary = target_component_summary(
                        direct_refs[m, k],
                        dout[k, m, k],
                        out[k, m, k],
                        out[:, m, k].sum(axis=0) - out[k, m, k],
                        nout[m, k],
                        mixref[m],
                    )
                    row["targets"].append(
                        {
                            "target": k,
                            "band": band,
                            "active_samples": int(m.sum()),
                            "input": input_summary,
                            "output": output_summary,
                        }
                    )
            rows.append(row)
        write_json(
            run / "metrics.json",
            {
                "status": "development_diagnostic_only",
                "completed_cases": index + 1,
                "expected_cases": len(cases),
                "run_status": "results_complete" if index + 1 == len(cases) else "in_progress",
                "rows": rows,
            },
        )
        print(f"Completed {case}", flush=True)
    write_json(
        run / "manifest.json",
        {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "config_sha256": canonical_hash(expanded),
            "input_config_sha256": sha256_file(config_path),
            "source_code_files_sha256": hashes,
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()
            },
        },
    )
    pending = run / "completion.pending.json"
    write_json(
        pending, {"run_status": "complete", "manifest_sha256": sha256_file(run / "manifest.json")}
    )
    pending.replace(run / "completion.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_component_replay(args.config, args.output_root))


if __name__ == "__main__":
    main()
