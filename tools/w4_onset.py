"""Frozen single/triple-source onset gates; only mixtures reach the estimators."""

import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.dsp.stft import StreamingSTFT
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.localization.music_reference import music_candidates
from acoustic_array.localization.onset import onset_filter
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room

try:
    from tools.w4_boundary import compose_subset, score_peaks, verify_run
except ModuleNotFoundError:
    from w4_boundary import compose_subset, score_peaks, verify_run


def main(config_path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = (config_path or root / "configs/experiments/w4_onset.json").resolve()
    cfg = load_json(config_path)
    parent = (config_path.parent / cfg["parent_run"]).resolve()
    baseline = (config_path.parent / cfg["baseline_run"]).resolve()
    evidence = verify_run(parent, require_completion=False)
    baseline_evidence = verify_run(baseline, require_completion=True)
    expanded = load_json(parent / "config.expanded.json")
    original = expanded["config"]
    previous = load_json(baseline / "config.expanded.json")
    if previous["parent"]["manifest_sha256"] != evidence["manifest_sha256"]:
        raise ValueError("Different baseline parent")
    for key in (
        "cases",
        "modes",
        "fft_size",
        "observation_seconds",
        "band_hz",
        "angle_grid_deg",
        "minimum_separation_deg",
        "matching_tolerance_deg",
    ):
        if cfg[key] != previous["config"][key]:
            raise ValueError(f"Baseline setting mismatch: {key}")
    if cfg["source_subsets"] not in ([[0], [1], [2]], [[0, 1, 2]]):
        raise ValueError("Only the preregistered single or triple source gates are allowed")
    old_rows = {
        (r["case"], r["mode"], tuple(r["source_subset"])): r
        for r in load_json(baseline / "metrics.json")["rows"]
    }
    array = ArrayConfig.from_mapping(expanded["array"])
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    stop = round(cfg["observation_seconds"] * array.sample_rate_hz)
    with np.load(parent / "sources.npz") as a:
        sources = a["sources"].copy()
    diagnoses = {r["case"]: r for r in json.loads((parent / "room_diagnostics.json").read_text())}
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    grid = np.arange(-90.0, 91.0)
    run = root / "artifacts/runs" / f"w4_onset_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    run.mkdir()
    write_json(
        run / "config.expanded.json",
        {
            "config": cfg,
            "array": expanded["array"],
            "parent": evidence,
            "baseline": baseline_evidence,
            "config_sha256": sha256_file(config_path),
            "scope": "known-source-count onset development gate",
        },
    )
    for source in [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__),
        root / "tools/w4_boundary.py",
    ]:
        dst = run / "source_snapshot" / source.resolve().relative_to(root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(source.read_bytes())
    rows = []
    for case in cfg["cases"]:
        d = diagnoses[case]
        with np.load(parent / f"{case}_reconstruction.npz") as a:
            rir = RoomRIR(
                a["rirs"],
                a["direct_rirs"],
                a["sources_m"],
                a["microphones_m"],
                a["direct_distances_m"],
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
                trim_samples=original["trim_samples"],
            )
            if not np.array_equal(built.source_gains, a["gains"]):
                raise ValueError("Gain replay mismatch")
        offset = rir.source_positions_m - np.asarray(original["room"]["array_center_m"])
        truth = np.rad2deg(np.arctan2(offset[:, 0], offset[:, 1]))
        saved = {"angles_deg": grid}
        for mode in cfg["modes"]:
            for ids in cfg["source_subsets"]:
                mixture = compose_subset(
                    built.scene.components,
                    direct * built.source_gains[:, None, None],
                    built.scene.noise,
                    tuple(ids),
                    mode,
                )
                frames = StreamingSTFT(stft, array.microphone_count).push(
                    mixture[:stop], sample_start=0
                )
                spectra = np.stack([f.spectrum for f in frames if f.sample_start >= 0])
                common = dict(
                    num_sources=len(ids),
                    min_frequency_hz=cfg["band_hz"][0],
                    max_frequency_hz=cfg["band_hz"][1],
                    minimum_separation_deg=cfg["minimum_separation_deg"],
                )
                control = music_candidates(spectra, array, frequencies, grid, **common)
                old = old_rows[(case, mode, tuple(ids))]
                if not np.allclose(
                    [[p.angle_deg, p.score] for p in control.peaks],
                    [[p["angle_deg"], p["score"]] for p in old["peaks"]],
                    atol=1e-8,
                    rtol=0,
                ):
                    raise ValueError("Baseline peak replay mismatch")
                control_score = score_peaks(
                    truth[ids], control.peaks, cfg["matching_tolerance_deg"]
                )
                if any(control_score[k] != old[k] for k in ("matched", "missed", "false_peaks")):
                    raise ValueError("Baseline scoring mismatch")
                filtered = onset_filter(spectra, **cfg["onset"])
                result = music_candidates(
                    filtered.filtered_spectra,
                    array,
                    frequencies,
                    grid,
                    frequency_mask=filtered.selected_counts > 0,
                    **common,
                )
                key = mode + "_" + "_".join(map(str, ids))
                saved[key + "_scores"] = result.scores
                saved[key + "_mask"] = filtered.mask
                saved[key + "_rise_db"] = filtered.rise_db
                rows.append(
                    {
                        "case": case,
                        "mode": mode,
                        "source_subset": ids,
                        "peaks": [asdict(p) for p in result.peaks],
                        "baseline_matched": old["matched"],
                        "frames": result.frame_count,
                        "retained_frequency_bins": result.frequency_bin_count,
                        "selected_counts": filtered.selected_counts.tolist(),
                        **score_peaks(truth[ids], result.peaks, cfg["matching_tolerance_deg"]),
                    }
                )
        np.savez_compressed(run / f"{case}_scores.npz", **saved)
        print(case, "complete", flush=True)
    summary = {
        mode: {
            k: sum(r[k] for r in rows if r["mode"] == mode)
            for k in ("matched", "missed", "false_peaks")
        }
        for mode in cfg["modes"]
    }
    gate = cfg["gate"]
    no_regression = all(r["matched"] >= r["baseline_matched"] for r in rows)
    passed = (
        summary["reverberant"]["matched"] >= gate["reverberant_minimum_matched"]
        and summary["reverberant"]["false_peaks"] <= gate["reverberant_maximum_false_peaks"]
        and summary["direct_only"]["matched"] >= gate["direct_minimum_matched"]
        and (no_regression or not gate["no_single_case_regression"])
    )
    write_json(
        run / "metrics.json",
        {
            "rows": rows,
            "summary": summary,
            "no_single_case_regression": no_regression,
            "decision_gate_passed": passed,
        },
    )
    write_json(
        run / "manifest.json",
        {
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {
                n: importlib.metadata.version(n) for n in ("numpy", "scipy", "pyroomacoustics")
            },
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p)
                for p in sorted(run.rglob("*"))
                if p.is_file()
            },
        },
    )
    tmp = run / "completion.json.tmp"
    write_json(tmp, {"manifest_sha256": sha256_file(run / "manifest.json")})
    tmp.replace(run / "completion.json")
    print(run, summary, "passed", passed, flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    main(parser.parse_args().config)
