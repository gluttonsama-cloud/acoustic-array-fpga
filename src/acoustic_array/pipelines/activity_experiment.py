"""Re-evaluate one saved speech mixture using fixed reference activity masks."""

import argparse
import importlib.metadata
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.core.config import ArrayConfig
from acoustic_array.evaluation.activity import ActivityConfig, masked_comparison, reference_activity
from acoustic_array.evaluation.metrics import bandpass_for_evaluation, comparison_metrics
from acoustic_array.evaluation.scene_statistics import summarize_s1
from acoustic_array.io.artifacts import load_json, sha256_file, write_json


def run_activity_experiment(parent: Path, output_root: Path) -> Path:
    parent = parent.resolve()
    manifest = load_json(parent / "manifest.json")

    def checked(name: str) -> Path:
        p = parent / name
        if sha256_file(p) != manifest["outputs_sha256"][name]:
            raise ValueError("Parent artifact hash mismatch")
        return p

    expanded = load_json(checked("config.expanded.json"))
    cfg = expanded["experiment"]
    array = ArrayConfig.from_mapping(expanded["array"])
    if len(cfg["fft_sizes"]) != 1:
        raise ValueError("Expected single FFT parent run")
    n = cfg["fft_sizes"][0]
    fs = array.sample_rate_hz
    mic = array.reference_microphone_index
    with np.load(checked("input.npz"), allow_pickle=False) as data:
        components, pcm = data["components"], data["mixture"]
    if (
        components.ndim != 3
        or components.shape[0] != 3
        or not components.size
        or components.shape[-1] != array.microphone_count
        or pcm.shape != components.shape[1:]
        or not np.isfinite(components).all()
        or not np.isfinite(pcm).all()
    ):
        raise ValueError("Expected three finite source components and matching mixture")
    reference = components[:, :, mic].T
    mixture = pcm[:, mic]
    ac = ActivityConfig()
    trim = round(cfg["evaluation_trim_s"] * fs)
    crop = slice(trim, -trim)
    if trim <= 0 or 2 * trim >= len(reference):
        raise ValueError("Invalid evaluation trim")
    masks = np.stack(
        [reference_activity(reference[:, k], fs, ac) for k in range(reference.shape[1])], axis=1
    )
    rb = bandpass_for_evaluation(reference, fs, tuple(cfg["evaluation_band_hz"]))
    mb = bandpass_for_evaluation(mixture, fs, tuple(cfg["evaluation_band_hz"]))
    rows = []
    stats = {}
    for method in ["das", "fixed_constraint"]:
        output = np.load(checked(f"{method}_{n}.npy"), allow_pickle=False)
        if output.shape != reference.shape:
            raise ValueError("Output/reference shapes differ")
        ob = bandpass_for_evaluation(output, fs, tuple(cfg["evaluation_band_hz"]))
        scores = []
        for k in range(reference.shape[1]):
            mask = masks[crop, k]
            row = {"method": method, "target": k, "active_fraction_after_trim": float(mask.mean())}
            for label, r, b, e in [
                ("fullband", reference, mixture, output),
                ("evaluation_band", rb, mb, ob),
            ]:
                row[label] = {
                    "whole": (
                        comparison_metrics(r[crop, k], b[crop], e[crop, k])
                        if np.ptp(r[crop, k]) > 0
                        else None
                    ),
                    "active": masked_comparison(r[crop, k], b[crop], e[crop, k], mask, fs, ac),
                }
            selected = row["evaluation_band"]["active"]
            scores.append(
                None if selected["metrics"] is None else selected["metrics"]["si_sdri_db"]
            )
            rows.append(row)
        ok = len(scores) == 3 and all(s is not None and np.isfinite(s) for s in scores)
        eligible = all(
            row["evaluation_band"]["active"]["status"] == "ok"
            for row in rows
            if row["method"] == method
        )
        stats[method] = summarize_s1(
            [
                {
                    "scene_id": parent.name,
                    "status": "ok" if ok else "failed",
                    "target_si_sdri_db": scores,
                }
            ]
        )
        if not eligible:
            stats[method] = {
                "status": "ineligible_reference_activity",
                "effect_scenes": 0,
                "reason": "not a declared silent test; insufficient/silent reference",
            }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"activity_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    np.save(run / "reference_masks.npy", masks)
    write_json(
        run / "metrics.json",
        {
            "status": "development_only_not_S1_or_B1",
            "activity_config": asdict(ac),
            "rows": rows,
            "scene_statistics": stats,
            "trim_samples": trim,
            "mask_reference": "unfiltered_fullband",
            "limitation": "parent used whole-clip RMS, not active-RMS scene generation",
        },
    )
    package = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "parent_run": str(parent),
            "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
            "expanded_parent": expanded,
            "created_utc": stamp,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "acoustic-array-fpga"]
                },
            },
            "argv": sys.argv,
            "source_code_files_sha256": {
                p.relative_to(package).as_posix(): sha256_file(p) for p in package.rglob("*.py")
            },
            "outputs_sha256": {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()},
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_activity_experiment(args.parent_run, args.output_root))


if __name__ == "__main__":
    main()
