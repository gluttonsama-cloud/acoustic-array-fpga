"""Fixed NormMUSIC source-count boundary diagnosis on the frozen W4 room scenes.

Run from the repository root: .venv/Scripts/python.exe tools/w4_boundary.py
All subsets use the same full-scene gains and calibrated noise. Source count is
known to the estimator; source geometry is used only after localization to score.
This is a development diagnosis, not a hardware or held-out acceptance result.
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
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.localization.music_reference import music_candidates
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room


def verify_run(run: Path, *, require_completion: bool) -> dict:
    """Check the seal when present and every indexed output, before using a run."""
    run = run.resolve()
    manifest = load_json(run / "manifest.json")
    manifest_hash = sha256_file(run / "manifest.json")
    completion_path = run / "completion.json"
    if completion_path.exists():
        completion = load_json(completion_path)
        if completion.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"Invalid completion seal: {run}")
    elif require_completion:
        raise ValueError(f"Missing completion seal: {run}")
    outputs = manifest.get("outputs_sha256")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"Missing output hashes: {run}")
    for name, digest in outputs.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Invalid archived output: {name}")
    return {"manifest_sha256": manifest_hash, "completion_present": completion_path.exists()}


def compose_subset(
    full_components: np.ndarray,
    direct_components: np.ndarray,
    noise: np.ndarray,
    subset: tuple[int, ...],
    mode: str,
) -> np.ndarray:
    """Sum frozen calibrated components and add one unchanged noise realization."""
    full = np.asarray(full_components)
    direct = np.asarray(direct_components)
    fixed_noise = np.asarray(noise)
    if (
        full.ndim != 3
        or full.shape != direct.shape
        or fixed_noise.shape != full.shape[1:]
        or not np.isfinite(full).all()
        or not np.isfinite(direct).all()
        or not np.isfinite(fixed_noise).all()
    ):
        raise ValueError("Invalid scene components")
    if mode not in ("direct_only", "reverberant"):
        raise ValueError("Unknown scene mode")
    if (
        not subset
        or len(set(subset)) != len(subset)
        or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= full.shape[0]
            for i in subset
        )
    ):
        raise ValueError("Invalid source subset")
    selected = direct if mode == "direct_only" else full
    return selected[list(subset)].sum(axis=0) + fixed_noise


def score_peaks(truth: np.ndarray, peaks: tuple, tolerance_deg: float) -> dict:
    estimates = np.asarray([p.angle_deg for p in peaks], dtype=float)
    if truth.ndim != 1 or not truth.size or not np.isfinite(truth).all():
        raise ValueError("Invalid scoring truth")
    if tolerance_deg <= 0:
        raise ValueError("Invalid matching tolerance")
    errors = np.abs(truth[:, None] - estimates[None, :])
    cost = np.where(errors <= tolerance_deg, errors, 1000.0)
    target_ids, peak_ids = linear_sum_assignment(cost)
    matched_errors = [
        float(errors[i, j])
        for i, j in zip(target_ids, peak_ids, strict=True)
        if errors[i, j] <= tolerance_deg
    ]
    return {
        "matched": len(matched_errors),
        "missed": len(truth) - len(matched_errors),
        "false_peaks": len(peaks) - len(matched_errors),
        "matched_mae_deg": float(np.mean(matched_errors)) if matched_errors else None,
        "matched_errors_deg": matched_errors,
    }


def check_baseline(row: dict, baseline_row: dict) -> None:
    """Require the reconstructed three-source reverberant control to replay W4."""
    if baseline_row.get("method") != "normmusic" or row.get("source_count_control") != 3:
        raise ValueError("Invalid W4 baseline comparison")
    if row["frames"] != baseline_row["frames"]:
        raise ValueError("W4 baseline frame count differs")
    old, new = baseline_row["peaks"], row["peaks"]
    if len(old) != len(new) or any(
        not np.isclose(a["angle_deg"], b["angle_deg"], rtol=0, atol=1e-8)
        or not np.isclose(a["score"], b["score"], rtol=0, atol=1e-8)
        for a, b in zip(old, new, strict=True)
    ):
        raise ValueError("W4 baseline peaks differ")
    for key in ("matched", "missed", "false_peaks"):
        if row[key] != baseline_row[key]:
            raise ValueError(f"W4 baseline {key} differs")


def _validate_config(cfg: dict, source_count: int, array: ArrayConfig) -> None:
    if cfg["modes"] != ["direct_only", "reverberant"]:
        raise ValueError("Expected both fixed scene modes")
    if cfg["source_subsets"] != [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]]:
        raise ValueError("Unexpected fixed source subsets")
    if source_count != 3 or array.microphone_count <= 3:
        raise ValueError("Expected the W4 three-source scene and at least four microphones")
    expected = {
        "observation_seconds": 3,
        "fft_size": 512,
        "band_hz": [500, 5000],
        "angle_grid_deg": [-90, 90, 1],
        "minimum_separation_deg": 10,
        "matching_tolerance_deg": 5,
    }
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f"Unexpected W4 control: {key}")
    if len(cfg["cases"]) != 4 or len(set(cfg["cases"])) != 4:
        raise ValueError("Expected four distinct W4 cases")


def main(config_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = (config_path or root / "configs/experiments/w4_boundary.json").resolve()
    cfg = load_json(config_path)
    parent = (config_path.parent / cfg["parent_run"]).resolve()
    baseline = (config_path.parent / cfg["baseline_run"]).resolve()
    parent_evidence = verify_run(parent, require_completion=False)  # Frozen G1 legacy run.
    baseline_evidence = verify_run(baseline, require_completion=True)
    expanded = load_json(parent / "config.expanded.json")
    baseline_expanded = load_json(baseline / "config.expanded.json")
    if baseline_expanded.get("parent_manifest_sha256") != parent_evidence["manifest_sha256"]:
        raise ValueError("W4 baseline was built from another parent archive")
    baseline_cfg = baseline_expanded["config"]
    for key in (
        "cases",
        "observation_seconds",
        "fft_size",
        "band_hz",
        "angle_grid_deg",
        "minimum_separation_deg",
        "matching_tolerance_deg",
    ):
        if baseline_cfg[key] != cfg[key]:
            raise ValueError(f"W4 baseline control differs: {key}")
    if (
        baseline_cfg["music"]["num_sources"] != 3
        or "normmusic" not in baseline_cfg["music"]["methods"]
    ):
        raise ValueError("W4 baseline NormMUSIC rank differs")
    original = expanded["config"]
    array = ArrayConfig.from_mapping(expanded["array"])
    with np.load(parent / "sources.npz") as archive:
        sources = archive["sources"].copy()
    _validate_config(cfg, sources.shape[0], array)
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    stop = round(cfg["observation_seconds"] * array.sample_rate_hz)
    if stop > sources.shape[1] or stop < stft.fft_size:
        raise ValueError("Invalid observation interval")
    diagnoses = {
        d["case"]: d
        for d in json.loads((parent / "room_diagnostics.json").read_text(encoding="utf-8"))
    }
    if not set(cfg["cases"]) <= diagnoses.keys():
        raise ValueError("Missing parent cases")
    baseline_rows = {
        r["case"]: r
        for r in load_json(baseline / "metrics.json")["rows"]
        if r["method"] == "normmusic"
    }
    if set(baseline_rows) != set(cfg["cases"]):
        raise ValueError("W4 baseline cases differ")
    low, high, step = cfg["angle_grid_deg"]
    grid = np.arange(low, high + step / 2, step)
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = root / "artifacts/runs" / f"w4_boundary_{stamp}"
    run.mkdir(exist_ok=False)
    write_json(
        run / "config.expanded.json",
        {
            "config": cfg,
            "array": expanded["array"],
            "config_sha256": sha256_file(config_path),
            "config_canonical_sha256": canonical_hash(cfg),
            "parent": parent_evidence,
            "baseline": baseline_evidence,
            "scope": (
                "development; same full-scene gains and noise; "
                "known source-count control; no hardware claim"
            ),
        },
    )
    snapshot = run / "source_snapshot"
    for source in [*sorted((root / "src").rglob("*.py")), Path(__file__).resolve()]:
        target = snapshot / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    rows = []
    references = []
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
        direct_scaled = direct * built.source_gains[:, None, None]
        full_scaled = built.scene.components
        noise = built.scene.noise
        ref = array.reference_microphone_index
        for i in range(len(sources)):
            references.append(
                {
                    "case": case,
                    "source_index": i,
                    "active_samples_first_3s": int(np.count_nonzero(built.masks[i, :stop])),
                    "activity_fraction_first_3s": float(np.mean(built.masks[i, :stop])),
                    "direct_reference_rms_first_3s": float(
                        np.sqrt(np.mean(direct_scaled[i, :stop, ref] ** 2))
                    ),
                    "full_reference_rms_first_3s": float(
                        np.sqrt(np.mean(full_scaled[i, :stop, ref] ** 2))
                    ),
                }
            )
        offsets = rirs.source_positions_m - np.asarray(original["room"]["array_center_m"])
        truth = np.rad2deg(np.arctan2(offsets[:, 0], offsets[:, 1]))
        saved = {"angles_deg": grid}
        for mode in cfg["modes"]:
            for subset_list in cfg["source_subsets"]:
                subset = tuple(subset_list)
                mixture = compose_subset(full_scaled, direct_scaled, noise, subset, mode)
                frames = StreamingSTFT(stft, array.microphone_count).push(
                    mixture[:stop], sample_start=0
                )
                spectra = np.stack([f.spectrum for f in frames if f.sample_start >= 0])
                result = music_candidates(
                    spectra,
                    array,
                    frequencies,
                    grid,
                    method="normmusic",
                    num_sources=len(subset),
                    min_frequency_hz=cfg["band_hz"][0],
                    max_frequency_hz=cfg["band_hz"][1],
                    minimum_separation_deg=cfg["minimum_separation_deg"],
                )
                subset_key = "_".join(map(str, subset))
                saved[f"{mode}_sources_{subset_key}_scores"] = result.scores
                selected_truth = truth[list(subset)]
                row = {
                    "case": case,
                    "mode": mode,
                    "source_subset": list(subset),
                    "source_count_control": len(subset),
                    "method": "normmusic",
                    "frames": result.frame_count,
                    "frequency_bins": result.frequency_bin_count,
                    "truth_deg_scoring_only": selected_truth.tolist(),
                    "peaks": [asdict(p) for p in result.peaks],
                    "actual_selected_peak_count": len(result.peaks),
                    **score_peaks(selected_truth, result.peaks, cfg["matching_tolerance_deg"]),
                }
                if mode == "reverberant" and subset == (0, 1, 2):
                    check_baseline(row, baseline_rows[case])
                    with np.load(baseline / f"{case}_spatial_scores.npz") as old_scores:
                        if not np.allclose(
                            result.scores, old_scores["normmusic_scores"], rtol=0, atol=1e-8
                        ):
                            raise ValueError("W4 baseline spatial spectrum differs")
                rows.append(row)
        np.savez_compressed(run / f"{case}_spatial_scores.npz", **saved)
        print(case, "complete", flush=True)
    write_json(
        run / "metrics.json",
        {
            "status": "development_complete",
            "source_count_is_known_control": True,
            "reference_channel_index": ref,
            "reference_interval_samples": [0, stop],
            "reference_rms_definition": "full first 3s, including inactive samples",
            "activity_mask_definition": (
                "full-scene direct reference, frozen before source-subset selection"
            ),
            "source_references": references,
            "rows": rows,
        },
    )
    dependencies = {
        name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pyroomacoustics")
    }
    write_json(
        run / "manifest.json",
        {
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
            "parent_manifest_sha256": parent_evidence["manifest_sha256"],
            "baseline_manifest_sha256": baseline_evidence["manifest_sha256"],
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p)
                for p in sorted(run.rglob("*"))
                if p.is_file()
            },
        },
    )
    completion_temp = run / "completion.json.tmp"
    write_json(completion_temp, {"manifest_sha256": sha256_file(run / "manifest.json")})
    completion_temp.replace(run / "completion.json")
    print(run, flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    main(parser.parse_args().config)
