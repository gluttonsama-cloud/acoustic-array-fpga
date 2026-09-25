"""Reject invalid geometry studies before creating an experiment archive."""

import json
from pathlib import Path

import numpy as np
import pytest

from acoustic_array.io.artifacts import load_json, write_json
from acoustic_array.pipelines.mpdr_experiment import run_mpdr_experiment


@pytest.fixture
def config_and_root(tmp_path: Path) -> tuple[dict, Path, Path]:
    project = Path(__file__).resolve().parents[1]
    cfg = load_json(project / "configs/experiments/g1_mpdr.json")
    parent = tmp_path / "parent"
    parent.mkdir()
    write_json(parent / "manifest.json", {"outputs_sha256": {}})
    write_json(
        parent / "config.expanded.json",
        {
            "config": {
                "distances_m": [2],
                "adjacent_angles_deg": [30],
                "rt60_targets_s": [0, 0.2],
                "trim_samples": 128,
            },
            "array": load_json(project / "configs/array/ula16_v1.json"),
        },
    )
    write_json(
        parent / "room_diagnostics.json",
        [{"case": "r2_gap30_rt0"}, {"case": "r2_gap30_rt0.2"}],
    )
    np.savez(parent / "sources.npz", sources=np.zeros((3, 12000)))
    cfg.update(room_run="parent", fft_size=128, calibration_seconds=0.01)
    output_root = tmp_path / "output"
    return cfg, tmp_path, output_root


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selected_cases": []}, "selected_cases"),
        ({"selected_cases": ["r2_gap30_rt0", "r2_gap30_rt0"]}, "selected_cases"),
        ({"selected_cases": ["r2_gap30_rt0.0"]}, "selected_cases"),
        ({"selected_cases": ["r2_gap30_rt0", "unknown"]}, "selected_cases"),
        ({"steering_error": {"angle_offset_deg": 1}}, "steering_error"),
        (
            {"steering_error": {"angle_offset_deg": 0, "distance_offset_m": 0, "extra": 1}},
            "steering_error",
        ),
        (
            {"steering_error": {"angle_offset_deg": float("nan"), "distance_offset_m": 0}},
            "angle_offset_deg",
        ),
        (
            {"steering_error": {"angle_offset_deg": 0, "distance_offset_m": -2}},
            "distance",
        ),
        (
            {"steering_error": {"angle_offset_deg": 61, "distance_offset_m": 0}},
            "front half-plane",
        ),
    ],
)
def test_invalid_selected_cases_and_geometry_rejected_before_archive(
    config_and_root, changes, message
):
    cfg, base, output_root = config_and_root
    cfg.update(changes)
    config_path = base / "config.json"
    if np.isnan(cfg.get("steering_error", {}).get("angle_offset_deg", 0)):
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
    else:
        write_json(config_path, cfg)
    with pytest.raises(ValueError, match=message):
        run_mpdr_experiment(config_path, output_root)
    assert not output_root.exists()
