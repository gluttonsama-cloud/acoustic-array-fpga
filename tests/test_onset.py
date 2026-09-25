"""Analytic checks for the batch onset filter."""

import numpy as np
import pytest

from acoustic_array.localization.onset import onset_filter


def _spectrum(levels: np.ndarray, microphones: int = 2) -> np.ndarray:
    return np.repeat(levels[:, :, None].astype(np.complex128), microphones, axis=2)


def test_silence_and_constant_spectra_select_nothing():
    for x in (np.zeros((40, 2, 2), complex), np.full((40, 2, 2), 2 + 3j)):
        result = onset_filter(x)
        assert result.mask.shape == (40, 2)
        np.testing.assert_array_equal(result.mask, False)
        np.testing.assert_array_equal(result.selected_counts, 0)
        np.testing.assert_array_equal(result.filtered_spectra, 0)
        assert np.isfinite(result.rise_db).all()


def test_current_frame_does_not_enter_history_and_decay_is_rejected():
    levels = np.ones((12, 1))
    levels[4:7] = 10
    result = onset_filter(_spectrum(levels), minimum_snapshots=1)
    assert result.rise_db[4, 0] == pytest.approx(20)
    assert result.rise_db[5, 0] == pytest.approx(15)
    assert result.rise_db[7, 0] < 0
    np.testing.assert_array_equal(result.mask[:4], False)
    assert result.mask[4, 0]
    assert not result.mask[7, 0]


def test_frequency_independence_and_relative_power_floor():
    levels = np.ones((40, 3))
    levels[4::2, 0] = 10
    levels[5::2, 1] = 8
    levels[:, 2] = 1e-8
    levels[4::2, 2] = 2e-8
    result = onset_filter(_spectrum(levels), minimum_snapshots=1)
    assert result.selected_counts.tolist() == [8, 8, 8]
    assert np.all(result.mask[4::2, 0][:8])
    assert np.all(result.mask[5::2, 1][:8])
    assert np.all(result.mask[4::2, 2][:8])


def test_top_k_prefers_largest_rises_then_earliest_frame():
    levels = np.ones((44, 1))
    levels[4::2] = 10
    levels[20] = 20
    result = onset_filter(_spectrum(levels), minimum_snapshots=1)
    # ceil((44 - 4) * 0.2) = 8; one stronger rise and seven earliest ties.
    assert result.selected_counts[0] == 8
    assert np.flatnonzero(result.mask[:, 0]).tolist() == [4, 6, 8, 10, 12, 14, 16, 20]


def test_minimum_snapshots_clears_entire_frequency():
    levels = np.ones((44, 2))
    levels[4::2, 0] = 10
    levels[4:8, 1] = 10
    result = onset_filter(_spectrum(levels))
    np.testing.assert_array_equal(result.mask, False)  # cap of eight is below default sixteen
    np.testing.assert_array_equal(result.selected_counts, 0)
    result = onset_filter(_spectrum(levels), minimum_snapshots=8)
    assert result.selected_counts.tolist() == [8, 0]


def test_relative_floor_uses_each_frequency_peak():
    levels = np.full((40, 1), 1e-3)
    levels[4::2] = 1e-2
    levels[20] = 10
    result = onset_filter(_spectrum(levels), relative_floor=1e-4, minimum_snapshots=1)
    assert np.flatnonzero(result.mask[:, 0]).tolist() == [20]


def test_common_extreme_gain_keeps_mask_and_output_is_input_or_zero():
    levels = np.ones((100, 2))
    levels[4::2, 0] = 10
    levels[5::2, 1] = 6
    x = _spectrum(levels) * (1 + 2j)
    original = x.copy()
    baseline = onset_filter(x)
    assert baseline.selected_counts.tolist() == [20, 20]
    for gain in (1e-200, 1e200):
        scaled = onset_filter(x * gain)
        np.testing.assert_array_equal(scaled.mask, baseline.mask)
        np.testing.assert_allclose(
            scaled.filtered_spectra, np.where(scaled.mask[:, :, None], x * gain, 0)
        )
    np.testing.assert_array_equal(
        baseline.filtered_spectra, np.where(baseline.mask[:, :, None], x, 0)
    )
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"history_frames": True},
        {"history_frames": 0},
        {"rise_db": 0},
        {"rise_db": np.nan},
        {"rise_db": "3"},
        {"keep_fraction": 0},
        {"keep_fraction": 1.1},
        {"relative_floor": -1},
        {"relative_floor": np.inf},
        {"minimum_snapshots": False},
        {"minimum_snapshots": 0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        onset_filter(np.ones((20, 1, 2), complex), **kwargs)


@pytest.mark.parametrize(
    "x",
    [
        np.ones((20, 2), complex),
        np.ones((20, 2, 2)),
        np.zeros((0, 2, 2), complex),
        np.full((20, 2, 2), np.nan + 0j),
        np.full((20, 2, 2), np.inf + 0j),
    ],
)
def test_invalid_spectra(x):
    with pytest.raises(ValueError):
        onset_filter(x)


def test_weak_frequency_onsets_survive_unrelated_strong_frequency():
    levels = np.ones((100, 1))
    levels[4::2, 0] = 10
    weak = _spectrum(levels) * 1e-7
    strong = np.ones_like(weak)
    alone = onset_filter(weak)
    together = onset_filter(np.concatenate([weak, strong], axis=1))
    assert alone.selected_counts[0] == 20
    np.testing.assert_array_equal(together.mask[:, 0], alone.mask[:, 0])
    np.testing.assert_allclose(together.rise_db[:, 0], alone.rise_db[:, 0])
    assert together.selected_counts[1] == 0


def test_subnormal_frequency_scale_does_not_overflow():
    levels = np.ones((100, 1))
    levels[4::2, 0] = 10
    x = np.concatenate([_spectrum(levels) * 1e-320, np.ones((100, 1, 2), complex) * 1e300], axis=1)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = onset_filter(x)
    assert np.isfinite(result.rise_db).all()
    assert result.selected_counts.tolist() == [20, 0]
