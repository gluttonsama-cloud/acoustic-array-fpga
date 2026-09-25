"""Bounded W4 development replay; frozen mixture prefix to static directions.

Run from the repository root: .venv/Scripts/python.exe tools/w4_static_demo.py
No source truth is passed to localization. This is not a held-out acceptance run.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.dsp.stft import StreamingSTFT
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.localization.scan import scan_directions, select_peaks
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room


def main(config_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (config_path or root / "configs/experiments/w4_static.json").resolve()
    cfg = load_json(config_path)
    parent = (config_path.parent / cfg["room_run"]).resolve()
    manifest = load_json(parent / "manifest.json")
    for name, digest in manifest["outputs_sha256"].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or sha256_file(path) != digest:
            raise ValueError("Invalid parent output archive")
    expanded = load_json(parent / "config.expanded.json")
    original = expanded["config"]
    array = ArrayConfig.from_mapping(expanded["array"])
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    stop = round(cfg["observation_seconds"] * array.sample_rate_hz)
    with np.load(parent / "sources.npz") as archive:
        sources = archive["sources"].copy()
    if stop > sources.shape[1] or stop < stft.fft_size:
        raise ValueError("Invalid observation interval")
    diagnoses = json.loads((parent / "room_diagnostics.json").read_text(encoding="utf-8"))
    diagnoses = {d["case"]: d for d in diagnoses}
    if len(set(cfg["cases"])) != len(cfg["cases"]) or not set(cfg["cases"]) <= diagnoses.keys():
        raise ValueError("Invalid selected parent cases")
    low, high, step = cfg["angle_grid_deg"]
    grid = np.arange(low, high + step / 2, step)
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_kind = (
        "w4_music"
        if "music" in cfg
        else "w4_subband"
        if "subband" in cfg
        else "w4_temporal"
        if "temporal" in cfg
        else "w4_static"
    )
    run = root / "artifacts/runs" / f"{run_kind}_{stamp}"
    run.mkdir(exist_ok=False)
    dependency_names = ["numpy", "scipy"]
    if "music" in cfg:
        dependency_names.append("pyroomacoustics")
    write_json(
        run / "config.expanded.json",
        {
            "config": cfg,
            "array": expanded["array"],
            "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
            "scope": (
                "development; far-field; fixed first 3s; exact source count control"
                if "music" in cfg
                else "development; far-field; fixed first 3s; source cap not known exact count"
            ),
        },
    )
    snapshot = run / "source_snapshot"
    for source in [*sorted((root / "src").rglob("*.py")), Path(__file__).resolve()]:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    rows = []
    for case in cfg["cases"]:
        diagnosis = diagnoses[case]
        with np.load(parent / f"{case}_reconstruction.npz") as archive:
            rirs = RoomRIR(
                archive["rirs"],
                archive["direct_rirs"],
                archive["sources_m"],
                archive["microphones_m"],
                archive["direct_distances_m"],
                diagnosis["target_rt60_s"],
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
                trim_samples=original["trim_samples"],
            )
            if not np.array_equal(built.source_gains, archive["gains"]):
                raise ValueError("Parent gain replay differs")
        analyzer = StreamingSTFT(stft, array.microphone_count)
        frames = analyzer.push(built.scene.mixture[:stop], sample_start=0)
        # Only complete non-padding frames: no flush/future samples in observation.
        spectra = np.stack([f.spectrum for f in frames if f.sample_start >= 0])
        # Source geometry is used here ONLY to score results, never to scan.
        offsets = rirs.source_positions_m - np.asarray(original["room"]["array_center_m"])
        truth = np.rad2deg(np.arctan2(offsets[:, 0], offsets[:, 1]))
        saved = {"angles_deg": grid}
        estimates_by_method = []
        for method, threshold in cfg["minimum_score"].items():
            scores = scan_directions(
                spectra,
                array,
                frequencies,
                grid,
                method=method,
                min_frequency_hz=cfg["band_hz"][0],
                max_frequency_hz=cfg["band_hz"][1],
                relative_floor=cfg["relative_floor"],
            )
            peaks = select_peaks(
                scores,
                grid,
                minimum_score=threshold,
                minimum_separation_deg=cfg["minimum_separation_deg"],
                max_sources=cfg["max_sources"],
            )
            saved[method] = scores
            estimates_by_method.append((method, peaks, len(spectra)))
        if "temporal" in cfg:
            from acoustic_array.localization.temporal import temporal_candidates

            result = temporal_candidates(
                spectra,
                array,
                frequencies,
                grid,
                min_frequency_hz=cfg["band_hz"][0],
                max_frequency_hz=cfg["band_hz"][1],
                relative_floor=cfg["relative_floor"],
                minimum_separation_deg=cfg["minimum_separation_deg"],
                max_sources=cfg["max_sources"],
                **cfg["temporal"],
            )
            saved["temporal_votes"] = result.scores
            saved["temporal_block_scores"] = result.block_scores
            estimates_by_method.append(("temporal", result.peaks, result.frame_count))
            # Same-duration control separates discarded-tail effects from algorithm effects.
            control = scan_directions(
                spectra[: result.frame_count],
                array,
                frequencies,
                grid,
                min_frequency_hz=cfg["band_hz"][0],
                max_frequency_hz=cfg["band_hz"][1],
                relative_floor=cfg["relative_floor"],
            )
            control_peaks = select_peaks(
                control,
                grid,
                minimum_score=cfg["minimum_score"]["srp_phat"],
                minimum_separation_deg=cfg["minimum_separation_deg"],
                max_sources=cfg["max_sources"],
            )
            saved["srp_phat_same_frames"] = control
            estimates_by_method.append(("srp_phat_same_frames", control_peaks, result.frame_count))
        if "subband" in cfg:
            from acoustic_array.localization.subband import subband_candidates

            result = subband_candidates(
                spectra,
                array,
                frequencies,
                grid,
                min_frequency_hz=cfg["band_hz"][0],
                max_frequency_hz=cfg["band_hz"][1],
                minimum_separation_deg=cfg["minimum_separation_deg"],
                max_sources=cfg["max_sources"],
                **cfg["subband"],
            )
            saved["subband_votes"] = result.scores
            saved["subband_scores"] = result.subband_scores
            estimates_by_method.append(("subband", result.peaks, result.frame_count))
            control = scan_directions(
                spectra[: result.frame_count],
                array,
                frequencies,
                grid,
                min_frequency_hz=cfg["band_hz"][0],
                max_frequency_hz=cfg["band_hz"][1],
                relative_floor=cfg["relative_floor"],
            )
            control_peaks = select_peaks(
                control,
                grid,
                minimum_score=cfg["minimum_score"]["srp_phat"],
                minimum_separation_deg=cfg["minimum_separation_deg"],
                max_sources=cfg["max_sources"],
            )
            saved["srp_phat_same_frames"] = control
            estimates_by_method.append(("srp_phat_same_frames", control_peaks, result.frame_count))
        if "music" in cfg:
            from acoustic_array.localization.music_reference import music_candidates

            music_cfg = cfg["music"]
            methods = music_cfg["methods"]
            if (
                not isinstance(methods, list)
                or not methods
                or len(methods) != len(set(methods))
                or music_cfg["primary_method"] not in methods
            ):
                raise ValueError("Invalid MUSIC method configuration")
            for method in methods:
                result = music_candidates(
                    spectra,
                    array,
                    frequencies,
                    grid,
                    method=method,
                    num_sources=music_cfg["num_sources"],
                    min_frequency_hz=cfg["band_hz"][0],
                    max_frequency_hz=cfg["band_hz"][1],
                    minimum_separation_deg=cfg["minimum_separation_deg"],
                )
                saved[f"{method}_scores"] = result.scores
                estimates_by_method.append((method, result.peaks, result.frame_count))
        for method, peaks, used_frames in estimates_by_method:
            estimates = np.array([p.angle_deg for p in peaks])
            errors = np.abs(truth[:, None] - estimates[None, :])
            tolerance = cfg["matching_tolerance_deg"]
            # Penalty dominates all possible in-gate errors: maximize matched count first.
            cost = np.where(errors <= tolerance, errors, 1000.0)
            target_ids, peak_ids = linear_sum_assignment(cost)
            matched_errors = [
                float(errors[i, j])
                for i, j in zip(target_ids, peak_ids, strict=True)
                if errors[i, j] <= tolerance
            ]
            rows.append(
                {
                    "case": case,
                    "method": method,
                    "frames": used_frames,
                    "truth_deg_scoring_only": truth.tolist(),
                    "peaks": [asdict(p) for p in peaks],
                    "matched": len(matched_errors),
                    "missed": len(truth) - len(matched_errors),
                    "false_peaks": len(peaks) - len(matched_errors),
                    "matched_mae_deg": float(np.mean(matched_errors)) if matched_errors else None,
                    "matched_errors_deg": matched_errors,
                }
            )
        np.savez_compressed(run / f"{case}_spatial_scores.npz", **saved)
        print(case, rows[-len(estimates_by_method) :], flush=True)
    metrics = {"status": "development_complete", "rows": rows}
    candidate_method = (
        cfg["music"]["primary_method"]
        if "music" in cfg
        else "subband"
        if "subband" in cfg
        else "temporal"
        if "temporal" in cfg
        else None
    )
    if candidate_method is not None:
        baseline = {r["case"]: r for r in rows if r["method"] == "srp_phat"}
        candidate = [r for r in rows if r["method"] == candidate_method]
        matched = sum(r["matched"] for r in candidate)
        false_peaks = sum(r["false_peaks"] for r in candidate)
        no_regression = all(r["matched"] >= baseline[r["case"]]["matched"] for r in candidate)
        gate = cfg["decision_gate"]
        metrics["decision"] = {
            "matched": matched,
            "false_peaks": false_peaks,
            "no_case_match_regression": no_regression,
            "pass_for_audio_integration": (
                matched >= gate["minimum_matched"]
                and false_peaks <= gate["maximum_false_peaks"]
                and (no_regression or not gate["no_case_match_regression"])
            ),
        }
    write_json(run / "metrics.json", metrics)
    write_json(
        run / "manifest.json",
        {
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {name: importlib.metadata.version(name) for name in dependency_names},
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p)
                for p in sorted(run.rglob("*"))
                if p.is_file()
            },
        },
    )
    write_json(run / "completion.json", {"manifest_sha256": sha256_file(run / "manifest.json")})
    print(run, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    main(parser.parse_args().config)
