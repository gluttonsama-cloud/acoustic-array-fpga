"""Calibrate MPDR on an early mixture prefix, score disjoint later room audio."""

import argparse
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
from acoustic_array.beamforming.mpdr import mixture_covariance, mpdr_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real
from acoustic_array.core.geometry import directions
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.pipelines.room_experiment import _score
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room


def run_mpdr_experiment(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    required = {
        "schema_version",
        "room_run",
        "fft_size",
        "calibration_seconds",
        "diagonal_loadings",
        "primary_loading",
        "band_hz",
        "min_wng_db",
        "max_condition",
    }
    optional = {"selected_cases", "steering_error"}
    if (
        not isinstance(cfg, dict)
        or not required <= set(cfg)
        or set(cfg) - required - optional
        or cfg["schema_version"] != "0.1"
    ):
        raise ValueError("Invalid MPDR config")
    error = cfg.get("steering_error", {"angle_offset_deg": 0.0, "distance_offset_m": 0.0})
    if not isinstance(error, dict) or set(error) != {"angle_offset_deg", "distance_offset_m"}:
        raise ValueError("Invalid steering_error")
    angle_offset = finite_real(error["angle_offset_deg"], "angle_offset_deg")
    distance_offset = finite_real(error["distance_offset_m"], "distance_offset_m")
    seconds = finite_real(cfg["calibration_seconds"], "calibration_seconds")
    loadings = cfg["diagonal_loadings"]
    if (
        not isinstance(loadings, list)
        or not loadings
        or len(loadings) > 4
        or any(finite_real(v, "loading") <= 0 for v in loadings)
        or len(set(loadings)) != len(loadings)
        or cfg["primary_loading"] not in loadings
        or seconds <= 0
    ):
        raise ValueError("Invalid calibration/loadings")
    parent = (config_path.parent / cfg["room_run"]).resolve()
    manifest = load_json(parent / "manifest.json")
    # Inputs are trusted only after checking the parent's frozen output manifest.
    for name, digest in manifest["outputs_sha256"].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or sha256_file(path) != digest:
            raise ValueError("Parent archive hash mismatch or escaped path")
    expanded_parent = load_json(parent / "config.expanded.json")
    original = expanded_parent["config"]
    array = ArrayConfig.from_mapping(expanded_parent["array"])
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    calibration_samples = round(seconds * array.sample_rate_hz)
    trim = original["trim_samples"]
    start = calibration_samples + trim
    with np.load(parent / "sources.npz") as z:
        sources = z["sources"].copy()
    if (
        calibration_samples < stft.fft_size
        or start >= sources.shape[1] - trim
        or trim < stft.fft_size
    ):
        raise ValueError("Calibration/guard leaves no valid evaluation interval")
    diagnostics = json.loads((parent / "room_diagnostics.json").read_text(encoding="utf-8"))
    expected = (
        len(original["distances_m"])
        * len(original["adjacent_angles_deg"])
        * len(original["rt60_targets_s"])
    )
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != expected
        or len({d["case"] for d in diagnostics}) != expected
    ):
        raise ValueError("Incomplete parent grid")
    parent_cases = {d["case"] for d in diagnostics}
    selected = cfg.get("selected_cases", [d["case"] for d in diagnostics])
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(case, str) for case in selected)
        or len(set(selected)) != len(selected)
        or not set(selected) <= parent_cases
    ):
        raise ValueError("selected_cases must be a nonempty unique subset of parent cases")
    cases = []
    for diagnosis in diagnostics:
        case = diagnosis["case"]
        if case not in selected:
            continue
        matches = [
            (d, g, rt)
            for d in original["distances_m"]
            for g in original["adjacent_angles_deg"]
            for rt in original["rt60_targets_s"]
            if f"r{d}_gap{g}_rt{rt}" == case
        ]
        if len(matches) != 1:
            raise ValueError("Unknown parent case")
        distance, gap, rt = matches[0]
        assumed_angles = [-gap + angle_offset, angle_offset, gap + angle_offset]
        assumed_distance = distance + distance_offset
        # Validate the full selected geometry before creating any output directory.
        directions(assumed_angles)
        if not np.isfinite(assumed_distance) or assumed_distance <= 0:
            raise ValueError("Estimated source distance must be finite and positive")
        das_weights(array, stft, assumed_angles, source_distances_m=[assumed_distance] * 3)
        cases.append((diagnosis, distance, gap, rt, assumed_angles, assumed_distance))
    expected = len(cases)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"g1_mpdr_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    expanded = {
        "config": cfg,
        "parent": expanded_parent,
        "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
        "calibration_interval_samples": [0, calibration_samples],
        "evaluation_interval_samples": [start, sources.shape[1] - trim],
        "selected_cases": [item[0]["case"] for item in cases],
        "steering_error": {"angle_offset_deg": angle_offset, "distance_offset_m": distance_offset},
        "assumed_geometry_by_case": {
            item[0]["case"]: {"angles_deg": item[4], "distance_m": item[5]} for item in cases
        },
        "policy": (
            "mixture-only prefix covariance; true direct geometry with common injected "
            "steering error; frozen weights"
        ),
    }
    write_json(run / "config.expanded.json", expanded)
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for p in package.rglob("*.py"):
        name = p.relative_to(package).as_posix()
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        hashes[name] = sha256_file(target)
    rows = []

    def checkpoint(completed: int) -> None:
        write_json(
            run / "metrics.json",
            {
                "status": "development_only_not_B2",
                "run_status": "results_complete" if completed == expected else "in_progress",
                "expected_cases": expected,
                "completed_cases": completed,
                "rows": rows,
            },
        )

    checkpoint(0)
    for index, (diagnosis, distance, gap, rt, assumed_angles, assumed_distance) in enumerate(cases):
        case = diagnosis["case"]
        with np.load(parent / f"{case}_reconstruction.npz") as z:
            rirs = RoomRIR(
                z["rirs"],
                z["direct_rirs"],
                z["sources_m"],
                z["microphones_m"],
                z["direct_distances_m"],
                rt,
                diagnosis["energy_absorption"],
            )
            base, direct = propagate_room(sources, array, rirs)
            built = normalize_active_scene(
                base,
                array,
                activity_components=direct,
                active_rms=original["active_rms"],
                noise_snr_db=original["noise_snr_db"],
                seed=original["seed"],
                trim_samples=trim,
            )
            if not np.array_equal(built.masks, z["masks"]) or not np.array_equal(
                built.source_gains, z["gains"]
            ):
                raise ValueError("Parent scene normalization changed")
        covariance, frames = mixture_covariance(
            built.scene.mixture[:calibration_samples], array, stft
        )
        np.savez_compressed(run / f"{case}_calibration.npz", covariance=covariance, frames=frames)
        if angle_offset == 0 and distance_offset == 0:
            with np.load(parent / f"r{distance}_gap{gap}_{stft.fft_size}_weights.npz") as weights:
                methods = [
                    ("das", weights["das"].copy()),
                    ("fixed_constraint", weights["constrained"].copy()),
                ]
        else:
            distances = [assumed_distance] * 3
            das = das_weights(array, stft, assumed_angles, source_distances_m=distances)
            constrained = constrained_weights(
                array,
                stft,
                assumed_angles,
                source_distances_m=distances,
                band_hz=tuple(original["constraint_band_hz"]),
                max_condition=original["max_condition"],
                min_white_noise_gain_db=original["min_wng_db"],
            )
            np.savez_compressed(
                run / f"{case}_baseline_weights.npz",
                das=das,
                constrained=constrained.weights,
                status=constrained.status,
                residual=constrained.constraint_residual,
                condition=constrained.condition_number,
                wng_db=constrained.white_noise_gain_db,
            )
            methods = [
                ("das", das),
                ("fixed_constraint", constrained.weights),
            ]
        for loading in loadings:
            result = mpdr_weights(
                covariance,
                array,
                stft,
                assumed_angles,
                source_distances_m=[assumed_distance] * 3,
                diagonal_loading=loading,
                band_hz=tuple(cfg["band_hz"]),
                max_condition=cfg["max_condition"],
                min_white_noise_gain_db=cfg["min_wng_db"],
            )
            name = f"mpdr_load{loading}"
            methods.append((name, result.weights))
            np.savez_compressed(
                run / f"{case}_{name}_weights.npz",
                weights=result.weights,
                status=result.status,
                residual=result.constraint_residual,
                condition=result.condition_number,
                wng_db=result.white_noise_gain_db,
            )
        ref = array.reference_microphone_index
        references = direct[:, :, ref].T * built.source_gains
        masks = built.masks.copy()
        masks[:, :start] = False
        for name, weights in methods:
            # Process only AFTER calibration; no early outputs computed with future-trained weights.
            tail = beamform_with_weights(
                built.scene.mixture[calibration_samples:],
                array,
                stft,
                weights,
                block_size=original["block_size"],
            )
            output = np.pad(tail, ((calibration_samples, 0), (0, 0)))
            row = {
                "case": case,
                "distance_m": distance,
                "adjacent_angle_deg": gap,
                "target_rt60_s": rt,
                "assumed_angles_deg": assumed_angles,
                "assumed_distance_m": assumed_distance,
                "steering_error": {
                    "angle_offset_deg": angle_offset,
                    "distance_offset_m": distance_offset,
                },
                "fft_size": stft.fft_size,
                "method": name,
                "calibration_frames": frames,
            }
            for label, reference in [
                ("direct_reference", references),
                ("reverberant_reference", built.scene.components[:, :, ref].T),
            ]:
                row[label] = _score(
                    reference, built.scene.mixture[:, ref], output, masks, array, trim
                )
            rows.append(row)
        checkpoint(index + 1)
        print(f"Completed {case}", flush=True)
    write_json(
        run / "manifest.json",
        {
            "argv": sys.argv,
            "config_sha256": canonical_hash(expanded),
            "input_config_sha256": sha256_file(config_path),
            "source_code_files_sha256": hashes,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    n: importlib.metadata.version(n)
                    for n in ["numpy", "scipy", "acoustic-array-fpga"]
                },
            },
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()
            },
        },
    )
    # Only this marker certifies the manifest and all results were written.
    # Atomic rename prevents a partial completion document from looking valid.
    pending = run / "completion.pending.json"
    write_json(
        pending,
        {
            "run_status": "complete",
            "completed_cases": expected,
            "manifest_sha256": sha256_file(run / "manifest.json"),
        },
    )
    pending.replace(run / "completion.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_mpdr_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
