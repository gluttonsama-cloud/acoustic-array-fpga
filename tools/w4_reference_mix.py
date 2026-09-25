"""One preregistered reference-mix trial using frozen floating-point audio."""

import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_array.beamforming.reference_mix import mix_reference
from acoustic_array.core.config import ArrayConfig
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.pipelines.room_experiment import _score
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room

try:
    from tools.w4_audio import CASES
    from tools.w4_boundary import verify_run
except ModuleNotFoundError:
    from w4_audio import CASES
    from w4_boundary import verify_run


def protection_gate(rows: list[dict], cfg: dict) -> dict:
    if len(rows) != 4 or {r["case"] for r in rows} != set(CASES):
        return {"passed": False, "status": "incomplete"}
    scores = {band: [] for band in ("fullband", "evaluation_band")}
    localized = {band: [] for band in scores}
    for row in rows:
        for ref in ("direct_reference", "reverberant_reference"):
            for band in scores:
                targets = row.get(ref, {}).get(band, {}).get("targets", [])
                if len(targets) != 3:
                    return {"passed": False, "status": "incomplete"}
                for k, target in enumerate(targets):
                    metrics = target.get("metrics")
                    if (
                        target.get("status") != "ok"
                        or target.get("active_samples", 0) <= 0
                        or not metrics
                        or any(
                            metrics.get(key) is None or not np.isfinite(metrics[key])
                            for key in ("input_si_sdr_db", "output_si_sdr_db", "si_sdri_db")
                        )
                    ):
                        return {"passed": False, "status": "invalid_target"}
                    if ref == "direct_reference":
                        value = metrics["si_sdri_db"]
                        scores[band].append(value)
                        if row["assignment_scoring_only"][k]["localized_within_tolerance"]:
                            localized[band].append(value)
    if any(not v for v in localized.values()):
        return {"passed": False, "status": "no_localized_targets"}
    worst = {k: float(min(v)) for k, v in scores.items()}
    medians = {k: float(np.median(v)) for k, v in localized.items()}
    safe = all(v >= cfg["minimum_target_improvement_db"] for v in worst.values())
    useful = all(v >= cfg["minimum_localized_median_improvement_db"] for v in medians.values())
    return {
        "passed": safe and useful,
        "status": "passed" if safe and useful else "threshold_not_met",
        "protection_passed": safe,
        "utility_passed": useful,
        "minimum_target_improvement_db": worst,
        "localized_median_improvement_db": medians,
        "localized_target_count": len(localized["fullband"]),
    }


def main() -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/experiments/w4_reference_mix.json"
    cfg = load_json(config_path)
    if (
        tuple(cfg["cases"]) != CASES
        or cfg["reference_fraction"] != 0.5
        or cfg["minimum_target_improvement_db"] != -1
        or cfg["minimum_localized_median_improvement_db"] != 3
    ):
        raise ValueError("Preregistered candidate changed")
    audio = (config_path.parent / cfg["parent_audio_run"]).resolve()
    audio_evidence = verify_run(audio, require_completion=True)
    if (
        audio_evidence["manifest_sha256"]
        != "58987746354b51fc04d34d42e27b4d9211d7060a6803170b16113b5269745f40"
    ):
        raise ValueError("Frozen audio archive differs")
    parent = root / "artifacts/runs/g1_room_20260923T171109_297143Z"
    parent_evidence = verify_run(parent, require_completion=False)
    if (
        load_json(audio / "manifest.json")["parent_manifest_sha256"]
        != parent_evidence["manifest_sha256"]
    ):
        raise ValueError("Parent chain mismatch")
    original_expanded = load_json(parent / "config.expanded.json")
    original = original_expanded["config"]
    array = ArrayConfig.from_mapping(original_expanded["array"])
    interval = load_json(audio / "config.expanded.json")
    calibration = interval["calibration_samples"][1]
    start, stop = interval["evaluation_samples"]
    trim = original["trim_samples"]
    old_rows = {
        r["case"]: r
        for r in load_json(audio / "metrics.json")["rows"]
        if r["branch"] == "onset" and r["method"] == "mpdr"
    }
    if set(old_rows) != set(CASES):
        raise ValueError("Missing onset output")
    with np.load(parent / "sources.npz") as z:
        sources = z["sources"].copy()
    diagnoses = {d["case"]: d for d in json.loads((parent / "room_diagnostics.json").read_text())}
    run = root / "artifacts/runs" / f"w4_reference_mix_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    run.mkdir()
    write_json(
        run / "config.expanded.json",
        {
            "config": cfg,
            "array": original_expanded["array"],
            "audio_evidence": audio_evidence,
            "room_evidence": parent_evidence,
            "calibration_samples": [0, calibration],
            "evaluation_samples": [start, stop],
        },
    )
    for file in [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__),
        root / "tools/w4_audio.py",
        root / "tools/w4_boundary.py",
    ]:
        target = run / "source_snapshot" / file.resolve().relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    rows = []
    for case in CASES:
        d = diagnoses[case]
        with np.load(parent / f"{case}_reconstruction.npz") as z:
            rir = RoomRIR(
                z["rirs"],
                z["direct_rirs"],
                z["sources_m"],
                z["microphones_m"],
                z["direct_distances_m"],
                d["target_rt60_s"],
                d["energy_absorption"],
            )
            base, direct = propagate_room(sources, array, rir)
            built = normalize_active_scene(
                base,
                array,
                activity_components=direct,
                active_rms=original["active_rms"],
                noise_snr_db=original["noise_snr_db"],
                seed=original["seed"],
                trim_samples=trim,
            )
            if not np.array_equal(built.source_gains, z["gains"]) or not np.array_equal(
                built.masks, z["masks"]
            ):
                raise ValueError("Normalization replay differs")
        reference = built.scene.mixture[:, array.reference_microphone_index]
        with np.load(audio / f"{case}_onset_mpdr.npz") as z:
            unprotected = z["output"].copy()
        protected = np.zeros_like(unprotected)
        protected[calibration:] = mix_reference(
            unprotected[calibration:], reference[calibration:], cfg["reference_fraction"]
        )
        links = old_rows[case]["assignment_scoring_only"]
        order = [link["port"] for link in links]
        masks = built.masks.copy()
        masks[:, :start] = False
        masks[:, stop:] = False
        references = {
            "direct_reference": direct[:, :, array.reference_microphone_index].T
            * built.source_gains,
            "reverberant_reference": built.scene.components[
                :, :, array.reference_microphone_index
            ].T,
        }
        row = {
            "case": case,
            "assignment_scoring_only": links,
            "port_angles_deg": old_rows[case]["port_angles_deg"],
            "active_samples_by_target": [int(m.sum()) for m in masks],
        }
        for name, target in references.items():
            replay = _score(target, reference, unprotected[:, order], masks, array, trim)
            if replay != old_rows[case][name]:
                raise ValueError("Original audio score does not replay exactly")
            row[name] = _score(target, reference, protected[:, order], masks, array, trim)
        gain = 0.95 / max(
            np.max(np.abs(reference)), np.max(np.abs(unprotected)), np.max(np.abs(protected))
        )
        row["listening_gain"] = float(gain)
        np.savez_compressed(
            run / f"{case}_audio.npz",
            protected=protected,
            reference=reference,
            unprotected=unprotected,
        )
        for method, signal in [
            ("reference", reference[:, None]),
            ("unprotected", unprotected),
            ("protected", protected),
        ]:
            for port in range(signal.shape[1]):
                sf.write(
                    run / f"{case}_{method}_port{port}.wav",
                    signal[:, port] * gain,
                    array.sample_rate_hz,
                    subtype="PCM_24",
                )
        rows.append(row)
        write_json(run / "metrics.json", {"status": "in_progress", "rows": rows})
        print(case, "complete", flush=True)
    gate = protection_gate(rows, cfg)
    write_json(run / "metrics.json", {"status": "development_complete", "rows": rows, "gate": gate})
    write_json(
        run / "manifest.json",
        {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "config_sha256": sha256_file(config_path),
            "dependencies": {
                n: importlib.metadata.version(n) for n in ("numpy", "scipy", "soundfile")
            },
            "outputs_sha256": {
                f.relative_to(run).as_posix(): sha256_file(f)
                for f in sorted(run.rglob("*"))
                if f.is_file()
            },
        },
    )
    pending = run / "completion.pending.json"
    write_json(pending, {"manifest_sha256": sha256_file(run / "manifest.json")})
    pending.replace(run / "completion.json")
    print(run, gate, flush=True)
    return run


if __name__ == "__main__":
    main()
