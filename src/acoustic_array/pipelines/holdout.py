"""Preregistered S1 batch orchestration over the shared active-scene numerical kernel."""

import argparse
import importlib.metadata
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.evaluation.scene_statistics import summarize_s1, summarize_s1_stratified
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.io.dataset_split import validate_splits
from acoustic_array.pipelines.active_scene_experiment import run_active_scene_experiment

PREREGISTRATION_SHA256 = "bf0f63ba997e07c1b936c15e0ba75e35c7286b8d78287a387dd90c9a0efdf700"


def validate_design(cfg: dict, corpus: dict) -> None:
    """This runner implements only the registered v1 design; reject semantic drift."""
    expected = {
        "schema_version": "s1-preregistration-1.0",
        "duration_s": 10,
        "source_start_s": 0,
        "active_rms": 0.04,
        "loader_preprocess_rms": 0.04,
        "noise_snr_db": 20,
        "trim_samples": 2400,
        "angles_deg": [-45, 0, 50],
        "block_size": 733,
        "seed_rule": "20260922 + scene index in frozen manifest order",
        "evaluation_band_hz": [500, 5000],
        "baselines": ["single_reference_microphone", "das_same_fft_and_input"],
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported preregistered design")
    for name, n in [("primary", 512), ("secondary", 1024)]:
        if cfg[name] != {
            "method": "fixed_constraint",
            "fft_size": n,
            "hop_size": n // 2,
            "constraint_band_hz": [100, 5000],
            "min_wng_db": -10,
            "max_condition": 10000,
        }:
            raise ValueError("Unsupported method configuration")
    stats = cfg["statistics"]
    expected_stats = {
        "scene_count": 30,
        "strata": ["test-clean", "dev-clean", "dev-other"],
        "scenes_per_stratum": 10,
        "quantile_method": "inverted_cdf",
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260922,
        "confidence": 0.95,
        "failed_scene_score": "negative_infinity",
        "b1_median_min_db": 10,
        "b1_p10_min_db": 6,
    }
    if any(stats.get(k) != v for k, v in expected_stats.items()):
        raise ValueError("Unsupported statistics configuration")
    files, scenes = corpus["files"], corpus["scenes"]
    if len(files) != 90 or len(scenes) != 30:
        raise ValueError("Expected 90 files and 30 scenes")
    for key in ("speaker_id", "utterance_id", "sha256"):
        if len({r[key] for r in files}) != 90:
            raise ValueError("Repeated corpus identity")
    if len({s["scene_id"] for s in scenes}) != 30:
        raise ValueError("Repeated scene identity")
    index = {r["utterance_id"]: r for r in files}
    used = []
    for scene in scenes:
        if scene["source_subset"] not in stats["strata"] or len(scene["sources"]) != 3:
            raise ValueError("Invalid scene stratum or size")
        for source, angle in zip(scene["sources"], cfg["angles_deg"], strict=True):
            record = index[source["utterance_id"]]
            if (
                source["speaker_id"] != record["speaker_id"]
                or record["source_subset"] != scene["source_subset"]
                or source["angle_degrees"] != angle
                or source["start_seconds"] != 0
                or source["duration_seconds"] != 10
            ):
                raise ValueError("Scene/source contract mismatch")
            used.append(source["utterance_id"])
    if len(set(used)) != 90:
        raise ValueError("Reused scene source")
    if any(sum(s["source_subset"] == k for s in scenes) != 10 for k in stats["strata"]):
        raise ValueError("Incorrect stratum scene counts")


def score_rows(rows: list[dict], fft_size: int, method: str) -> list[float]:
    """Only three valid finite primary metrics make a successful scene."""
    selected = [r for r in rows if r["fft_size"] == fft_size and r["method"] == method]
    if len(selected) != 3 or {r["target"] for r in selected} != {0, 1, 2}:
        raise ValueError("Missing or duplicated required output")
    scores = []
    for row in sorted(selected, key=lambda r: r["target"]):
        active = row["evaluation_band"]["active"]
        if active["status"] != "ok":
            raise ValueError("Unplanned inactive source")
        metric = active["metrics"]
        value = metric["si_sdri_db"]
        if (
            metric["metric_status"]["si_sdri_db"] != "finite"
            or value is None
            or not np.isfinite(value)
        ):
            raise ValueError("Nonfinite required score")
        scores.append(float(value))
    return scores


def run_holdout(config_path: Path, output_root: Path, project: Path) -> Path:
    """No retry, source replacement, score-based selection or overwrite."""
    config_path, project = config_path.resolve(), project.resolve()
    if sha256_file(config_path) != PREREGISTRATION_SHA256:
        raise ValueError("Frozen preregistration hash mismatch")
    cfg = load_json(config_path)
    corpus_path = (config_path.parent / cfg["corpus_manifest"]).resolve()
    array_path = (config_path.parent / cfg["array_config"]).resolve()
    if sha256_file(corpus_path) != cfg["corpus_sha256"]:
        raise ValueError("Frozen corpus hash mismatch")
    if sha256_file(array_path) != cfg["array_sha256"]:
        raise ValueError("Frozen array hash mismatch")
    corpus = load_json(corpus_path)
    validate_design(cfg, corpus)
    registry_path = project / "data/manifests/split_registry.json"
    registry = load_json(registry_path)
    validate_splits(
        registry["records"]
        + [
            {
                "dataset_id": corpus["dataset_id"],
                "speaker_id": r["speaker_id"],
                "sha256": r["sha256"],
                "split": "heldout",
            }
            for r in corpus["files"]
        ]
    )
    index = {r["utterance_id"]: r for r in corpus["files"]}
    for r in index.values():
        (project / r["path"]).resolve().relative_to(project / "data/raw")
    run = output_root.resolve() / datetime.now(UTC).strftime("s1_%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "preregistration.json", cfg)
    write_json(run / "corpus.json", corpus)
    write_json(run / "registry.json", registry)
    methods = {
        "primary": (512, "fixed_constraint"),
        "secondary": (1024, "fixed_constraint"),
        "das_512": (512, "das"),
        "das_1024": (1024, "das"),
    }
    records = {name: [] for name in methods}
    for i, scene in enumerate(corpus["scenes"]):
        folder = run / f"scene_{i:02d}"
        folder.mkdir()
        speech = {
            "license": corpus["license"],
            "files": [
                {
                    **index[s["utterance_id"]],
                    "path": os.path.relpath(project / index[s["utterance_id"]]["path"], folder),
                }
                for s in scene["sources"]
            ],
        }
        write_json(folder / "speech.json", speech)
        local_cfg = {
            "schema_version": "0.1",
            "array_config": str(array_path),
            "speech_manifest": "speech.json",
            "duration_s": cfg["duration_s"],
            "seed": 20260922 + i,
            "angles_deg": cfg["angles_deg"],
            "active_rms": cfg["active_rms"],
            "noise_snr_db": cfg["noise_snr_db"],
            "trim_samples": cfg["trim_samples"],
            "fft_sizes": [512, 1024],
            "block_size": cfg["block_size"],
            "constraint_band_hz": [100, 5000],
            "min_wng_db": -10,
            "max_condition": 10000,
        }
        write_json(folder / "config.json", local_cfg)
        common = {"scene_id": scene["scene_id"], "stratum": scene["source_subset"]}
        try:
            child = run_active_scene_experiment(folder / "config.json", folder)
            metrics = load_json(child / "metrics.json")
            for name, (fft, method) in methods.items():
                try:
                    scores = score_rows(metrics["rows"], fft, method)
                    row = {**common, "status": "ok", "target_si_sdri_db": scores}
                except (ValueError, KeyError, TypeError) as error:
                    row = {**common, "status": "failed", "error": repr(error)}
                records[name].append({**row, "child_run": child.relative_to(run).as_posix()})
        except Exception as error:
            # Includes missing/corrupt/silent source. All planned denominators remain intact.
            for name in methods:
                records[name].append({**common, "status": "failed", "error": repr(error)})
        write_json(folder / "outcome.json", {k: v[-1] for k, v in records.items()})
        print(f"{i + 1}/30 {scene['scene_id']}: {records['primary'][-1]['status']}", flush=True)
    stats = cfg["statistics"]
    summary = {}
    for name, rows in records.items():
        summary[name] = summarize_s1_stratified(
            rows,
            {s: 10 for s in stats["strata"]},
            bootstrap_samples=stats["bootstrap_samples"],
            seed=stats["bootstrap_seed"],
        )
        summary[name]["by_stratum"] = {
            s: summarize_s1(
                [r for r in rows if r["stratum"] == s],
                bootstrap_samples=stats["bootstrap_samples"],
                seed=stats["bootstrap_seed"],
            )
            for s in stats["strata"]
        }
    primary = summary["primary"]
    passed = all(
        primary[k]["status"] == "finite" and primary[k]["db"] >= limit
        for k, limit in [("median", 10), ("p10", 6)]
    )
    write_json(
        run / "results.json",
        {
            "records": records,
            "summary": summary,
            "b1_effect_thresholds_met": passed,
            "scope": "S1 given-DOA simulated heldout only; not B2/fixed-point/hardware acceptance",
            "child_status_note": (
                "Shared single-scene kernel retains development label; parent supplies "
                "frozen holdout selection, failure retention and statistics."
            ),
        },
    )
    package = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "status": "completed",
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "argv": sys.argv,
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "soundfile", "acoustic-array-fpga"]
                },
            },
            "preregistration_sha256": sha256_file(config_path),
            "inputs_sha256": {
                str(p): sha256_file(p) for p in [corpus_path, array_path, registry_path]
            },
            "source_code_sha256": {
                p.relative_to(package).as_posix(): sha256_file(p) for p in package.rglob("*.py")
            },
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()
            },
        },
    )
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_holdout(args.config, args.output_root, Path.cwd()))
