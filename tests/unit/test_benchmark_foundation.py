import numpy as np
import pytest

from acoustic_array.evaluation.activity import ActivityConfig, masked_comparison, reference_activity
from acoustic_array.evaluation.scene_statistics import summarize_s1
from acoustic_array.io.dataset_split import validate_splits


def test_activity_exact_frame_and_hangover():
    x = np.zeros(100)
    x[:5] = 1
    x[5:10] = -1
    cfg = ActivityConfig(frame_ms=10, hop_ms=10, hangover_ms=20)
    np.testing.assert_array_equal(reference_activity(x, 1000, cfg), np.arange(100) < 30)
    np.testing.assert_array_equal(reference_activity(x + 3, 1000, cfg), np.arange(100) < 30)


@pytest.mark.parametrize("x", [np.zeros(100), np.ones(100), np.arange(100) * 1e-9])
def test_silent_or_below_floor(x):
    assert not reference_activity(x, 1000).any()


def test_masked_analytic_sdr():
    t = np.arange(1000)
    ref = np.sin(2 * np.pi * t / 100)
    e = 0.1 * np.cos(2 * np.pi * t / 100)
    mask = np.ones(1000, dtype=bool)
    r = masked_comparison(ref, ref + e, ref + e / 10, mask, 1000)
    assert r["metrics"]["si_sdri_db"] == pytest.approx(20)
    assert (
        masked_comparison(ref, ref, ref, np.zeros(1000, dtype=bool), 1000)["status"]
        == "insufficient_activity"
    )
    with pytest.raises(ValueError):
        masked_comparison(ref, ref, ref, np.ones(1000), 1000)


def test_insufficient_tail_and_short_signal():
    mask = reference_activity([1, -1], 1000)
    assert mask.shape == (2,)
    assert masked_comparison([1, -1], [1, -1], [1, -1], mask, 1000)["metrics"] is None


@pytest.mark.parametrize(
    "kwargs", [{"hop_ms": 30}, {"absolute_rms": 0}, {"relative_db": 1}, {"frame_ms": float("nan")}]
)
def test_bad_activity_config(kwargs):
    with pytest.raises(ValueError):
        ActivityConfig(**kwargs)


def test_scene_failure_is_not_dropped_and_reproducible():
    records = [
        {"scene_id": str(i), "status": "ok", "target_si_sdri_db": [i, i + 1, i + 2]}
        for i in range(1, 10)
    ]
    records += [
        {"scene_id": "failure", "status": "failed"},
        {"scene_id": "silence", "status": "silent"},
    ]
    r = summarize_s1(records, bootstrap_samples=100, seed=2)
    assert r == summarize_s1(records, bootstrap_samples=100, seed=2)
    assert r["effect_scenes"] == 10 and r["silent_scenes"] == 1
    assert r["median"]["db"] == 4 and r["p10"]["status"] == "negative_infinity"
    assert r["failure_fraction"] == 0.1 and r["positive_fraction"] == 0.9


def test_all_failed_and_no_effect():
    r = summarize_s1([{"scene_id": "x", "status": "failed"}], bootstrap_samples=5)
    assert r["median_ci95"][0]["status"] == "negative_infinity"
    assert summarize_s1([])["status"] == "no_effect_scenes"


def test_duplicate_scene_and_bad_scores_rejected():
    row = {"scene_id": "x", "status": "ok", "target_si_sdri_db": [1, 2, 3]}
    with pytest.raises(ValueError):
        summarize_s1([row, row])
    row["target_si_sdri_db"] = [1, np.nan, 3]
    with pytest.raises(ValueError):
        summarize_s1([row])


def record(speaker, split, digest):
    return {"dataset_id": "libri", "speaker_id": speaker, "split": split, "sha256": digest * 64}


def test_split_leakage_detection():
    assert not validate_splits([record("1", "development", "a")])["has_heldout"]
    assert validate_splits([record("1", "development", "a"), record("2", "heldout", "b")])[
        "has_heldout"
    ]
    with pytest.raises(ValueError, match="Speaker leakage"):
        validate_splits([record("1", "development", "a"), record("1", "heldout", "b")])
    with pytest.raises(ValueError, match="Source file leakage"):
        validate_splits([record("1", "development", "a"), record("2", "heldout", "a")])
