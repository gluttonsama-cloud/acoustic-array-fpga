"""Checks for the frozen weak-event input and physical calibration contract."""

from pathlib import Path

import numpy as np
import pytest

from acoustic_array.pipelines.weak_knock_experiment import (
    fixed_noise,
    load_inputs,
    scene_for_case,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/experiments/w1_knock.json"
ROOT = CONFIG.parents[2]
PARENT = ROOT / "artifacts/runs/g1_room_20260923T042446_546511Z/manifest.json"
KNOCK = ROOT / "data/external/esc50_knock/1-101336-A-30.wav"


@pytest.mark.skipif(
    not PARENT.is_file() or not KNOCK.is_file(),
    reason="Pinned parent room archive and licensed knock recording are local development inputs",
)
def test_real_pinned_inputs_target_absent_during_training():
    cfg, room, manifest, array, sources, knock = load_inputs(CONFIG)
    assert manifest["file"]["metadata"]["category"] == "door_wood_knock"
    assert len(knock) == 5 * array.sample_rate_hz
    target, other, direct, meta = scene_for_case(cfg, room, cfg["cases"][0], array, sources, knock)
    mask = meta["mask"]
    ref = array.reference_microphone_index
    assert mask.dtype == np.bool_
    assert not mask[: round(3.65 * array.sample_rate_hz)].any()
    assert not mask[round(9.35 * array.sample_rate_hz) :].any()
    assert np.max(np.abs(target[: 4 * array.sample_rate_hz])) == 0
    assert np.max(np.abs(direct[: 4 * array.sample_rate_hz])) == 0
    assert np.mean(target[mask, ref] ** 2) ** 0.5 == pytest.approx(0.01)
    assert meta["calibrated_interference_on_sir_db"] == pytest.approx(-10)
    assert np.any(other[: 3 * array.sample_rate_hz])


def test_frozen_grid_has_36_unique_method_conditions():
    from acoustic_array.io.artifacts import load_json

    cfg = load_json(CONFIG)
    keys = {
        (case, on, error, method)
        for case in cfg["cases"]
        for on in (True, False)
        for error in cfg["angle_errors_deg"]
        for method in ("das", "loaded_single", "sector_lcmv")
    }
    assert len(keys) == 36


def test_noise_is_repeatable_and_exactly_calibrated_on_same_mask():
    target = np.tile(np.arange(1, 9, dtype=float)[:, None], (1, 3))
    mask = np.array([False, True, True, True, False, True, False, False])
    first = fixed_noise(target, mask, 1, 20, 1234)
    np.testing.assert_array_equal(first, fixed_noise(target, mask, 1, 20, 1234))
    assert 10 * np.log10(
        np.mean(target[mask, 1] ** 2) / np.mean(first[mask, 1] ** 2)
    ) == pytest.approx(20)
