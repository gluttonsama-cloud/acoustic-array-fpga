"""Analytic and adversarial checks for bounded temporal SRP voting."""

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.localization.temporal import (
    _prominent_block_peaks,
    temporal_candidates,
)

ARRAY = ArrayConfig(48000, 16, 0.03048, 7)
FREQ = np.fft.rfftfreq(512, 1 / 48000)
GRID = np.arange(-90.0, 91.0)


def plane_wave(angle: float, frames: int = 20, array: ArrayConfig = ARRAY) -> np.ndarray:
    # Analytic travel phase, independent of the steering implementation.
    if array.positions_m is None:
        positions = np.zeros((array.microphone_count, 3))
        positions[:, 0] = (
            np.arange(array.microphone_count) - (array.microphone_count - 1) / 2
        ) * array.spacing_m
    else:
        positions = np.asarray(array.positions_m)
    direction = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0])
    delay = -(positions @ direction) / 343.0
    spectrum = np.exp(-2j * np.pi * FREQ[:, None] * delay)
    return np.broadcast_to(spectrum, (frames, *spectrum.shape)).copy()


@pytest.mark.parametrize("angle", [-35.0, 0.0, 28.0])
def test_analytic_single_source_sign_and_gain(angle):
    x = plane_wave(angle)
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.block_count == 1 and result.frame_count == 20
    assert result.peaks[0].angle_deg == angle
    assert result.peaks[0].score == pytest.approx(1)
    assert len(result.peaks) == 1
    np.testing.assert_allclose(
        result.scores, temporal_candidates(x * 1e-200, ARRAY, FREQ, GRID).scores
    )
    np.testing.assert_allclose(
        result.scores, temporal_candidates(x * 1e200, ARRAY, FREQ, GRID).scores
    )


@pytest.mark.parametrize("angle", [-90.0, 90.0])
def test_true_endfire_peak_can_use_one_sided_prominence(angle):
    result = temporal_candidates(plane_wave(angle), ARRAY, FREQ, GRID)
    assert any(peak.angle_deg == angle for peak in result.peaks)


@pytest.mark.parametrize("angle", [-90.0, 90.0])
def test_staggered_geometry_endfire(angle):
    x = (np.arange(16) - 7.5) * 0.03048
    y = np.tile([-0.03, 0.03], 8)
    positions = tuple((float(xx), float(yy), 0.0) for xx, yy in zip(x, y, strict=True))
    array = ArrayConfig(48000, 16, 0.03048, 7, positions_m=positions)
    result = temporal_candidates(plane_wave(angle, array=array), array, FREQ, GRID)
    assert any(peak.angle_deg == angle for peak in result.peaks)


def test_silence_and_excluded_dc_nyquist():
    x = np.zeros((40, len(FREQ), ARRAY.microphone_count), complex)
    x[:, [0, -1]] = 1
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.block_count == 2 and result.frame_count == 40
    np.testing.assert_array_equal(result.block_scores, 0)
    np.testing.assert_array_equal(result.scores, 0)
    assert result.peaks == ()


def test_intermittent_alternating_sources_and_incomplete_tail():
    x = np.concatenate([plane_wave(-30), plane_wave(30), np.zeros_like(plane_wave(0, 7))])
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.block_count == 2 and result.frame_count == 40
    assert sorted(peak.angle_deg for peak in result.peaks) == [-30, 30]
    assert all(peak.score == pytest.approx(0.5) for peak in result.peaks)
    np.testing.assert_array_equal(
        result.scores, temporal_candidates(x[:40], ARRAY, FREQ, GRID).scores
    )


def test_silent_block_is_in_support_denominator():
    x = np.concatenate([plane_wave(23), np.zeros_like(plane_wave(0))])
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.peaks[0].angle_deg == 23
    assert result.peaks[0].score == pytest.approx(0.5)


def test_independent_noise_has_no_supported_candidate():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(100, len(FREQ), 16)) + 1j * rng.normal(size=(100, len(FREQ), 16))
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.peaks == ()


def test_flat_map_and_duplicate_block_votes():
    assert _prominent_block_peaks(np.full(5, 0.8), np.arange(5.0), 0.1, 0.05, 10, 30, 0, 3) == ()
    x = np.concatenate([plane_wave(0), plane_wave(0)])
    result = temporal_candidates(x, ARRAY, FREQ, GRID)
    assert result.scores.max() == pytest.approx(1)
    assert np.all(result.scores <= 1)
    assert len(result.peaks) == 1


def test_per_frame_phat_equalizes_alternating_gain_and_caps_overlapping_votes():
    # Static pre-averaging would let the high-gain half dominate this block.
    x = np.concatenate([plane_wave(-20, 10) * 1e4, plane_wave(20, 10)])
    result = temporal_candidates(x, ARRAY, FREQ, GRID, vote_radius_deg=30)
    assert sorted(peak.angle_deg for peak in result.peaks) == [-20, 20]
    assert result.scores[90] == pytest.approx(1 / 3)  # max of two triangular votes
    assert result.scores.max() == pytest.approx(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_frames": True},
        {"block_frames": 0},
        {"max_sources": True},
        {"max_sources": 0},
        {"minimum_score": np.nan},
        {"minimum_prominence": -1},
        {"prominence_radius_deg": 0},
        {"boundary_prominence_radius_deg": 0},
        {"boundary_prominence_radius_deg": 9},
        {"vote_radius_deg": 0},
        {"minimum_support": 1.1},
        {"minimum_separation_deg": -1},
        {"relative_floor": 1},
        {"min_frequency_hz": 5000},
        {"max_frequency_hz": np.inf},
    ],
)
def test_malformed_parameters(kwargs):
    with pytest.raises(ValueError):
        temporal_candidates(plane_wave(0), ARRAY, FREQ, GRID, **kwargs)


def test_malformed_input_arrays_and_short_valid_input():
    x = plane_wave(0)
    bad_inputs = [x[0], x.real, x[:, :, :1], np.full_like(x, np.nan)]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            temporal_candidates(bad, ARRAY, FREQ, GRID)
    with pytest.raises(ValueError):
        temporal_candidates(x, ARRAY, FREQ[::-1], GRID)
    with pytest.raises(ValueError):
        temporal_candidates(x, ARRAY, FREQ, GRID[::-1])
    with pytest.raises(ValueError):
        temporal_candidates(x, ARRAY, FREQ, np.array([False, True]))
    result = temporal_candidates(x[:19], ARRAY, FREQ, GRID)
    assert result.block_scores.shape == (0, len(GRID))
    assert result.frame_count == 0 and result.peaks == ()
