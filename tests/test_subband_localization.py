"""Independent tests for time-frequency dominant-direction histograms."""

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.localization.subband import subband_candidates

ARRAY = ArrayConfig(48000, 16, 0.03048, 7)
FREQ = np.fft.rfftfreq(512, 1 / 48000)
GRID = np.arange(-90.0, 91.0)


def plane_wave(angle: float, array: ArrayConfig = ARRAY) -> np.ndarray:
    """Analytic travel phase independent of production steering code."""
    if array.positions_m is None:
        positions = np.zeros((array.microphone_count, 3))
        positions[:, 0] = (
            np.arange(array.microphone_count) - (array.microphone_count - 1) / 2
        ) * array.spacing_m
    else:
        positions = np.asarray(array.positions_m)
    radians = np.deg2rad(angle)
    path = -positions[:, 0] * np.sin(radians) - positions[:, 1] * np.cos(radians)
    return np.exp(-2j * np.pi * FREQ[:, None] * path / 343.0)


def disjoint_scene(angles: list[float], frames: int = 40) -> np.ndarray:
    result = np.zeros((frames, len(FREQ), ARRAY.microphone_count), complex)
    bins = np.flatnonzero((FREQ >= 500) & (FREQ <= 5000))
    for group, start in enumerate(range(0, len(bins), 4)):
        selected = bins[start : start + 4]
        wave = plane_wave(angles[group % len(angles)])
        result[:, selected] = wave[selected]
    return result


@pytest.mark.parametrize("angle", [-35.0, 0.0, 28.0])
def test_single_source_sign_and_extreme_common_gain(angle):
    scene = disjoint_scene([angle], 20)
    result = subband_candidates(scene, ARRAY, FREQ, GRID)
    assert result.block_count == 1
    assert result.frame_count == 20
    assert result.subband_count == 12
    assert result.accepted_count == 12
    assert result.peaks[0].angle_deg == angle
    assert result.peaks[0].score == pytest.approx(1)
    np.testing.assert_allclose(
        result.scores, subband_candidates(scene * 1e-200, ARRAY, FREQ, GRID).scores
    )
    np.testing.assert_allclose(
        result.scores, subband_candidates(scene * 1e200, ARRAY, FREQ, GRID).scores
    )


def test_three_sources_in_disjoint_subbands_with_unequal_gain():
    scene = disjoint_scene([-30, 0, 30], 40)
    bins = np.flatnonzero((FREQ >= 500) & (FREQ <= 5000))
    for group, start in enumerate(range(0, len(bins), 4)):
        scene[:, bins[start : start + 4]] *= 10.0 ** (group % 3 - 1)
    result = subband_candidates(scene, ARRAY, FREQ, GRID)
    assert sorted(peak.angle_deg for peak in result.peaks) == [-30, 0, 30]
    assert all(peak.score == pytest.approx(1 / 3) for peak in result.peaks)
    assert np.all(result.scores <= 1)


def test_energy_gate_preserves_physical_ratio_across_different_frames():
    spectra = np.zeros((20, len(FREQ), 16), complex)
    bins = np.flatnonzero((FREQ >= 500) & (FREQ <= 5000))
    # Use resolved mid-band groups so this isolates energy gating, not low-band aperture.
    strong, weak = bins[20:24], bins[24:28]
    left, right = plane_wave(-30), plane_wave(30)
    spectra[:10, strong] = left[strong]
    spectra[10:, weak] = weak_gain = right[weak] * 1e-4
    assert np.mean(np.abs(weak_gain) ** 2) / np.mean(np.abs(left[strong]) ** 2) < 1e-5
    result = subband_candidates(spectra, ARRAY, FREQ, GRID)
    assert [peak.angle_deg for peak in result.peaks] == [-30]
    assert result.accepted_count == 1


@pytest.mark.parametrize("angle", [-90.0, 90.0])
def test_actual_array_endfire_survives_boundary_prominence(angle):
    result = subband_candidates(disjoint_scene([angle], 20), ARRAY, FREQ, GRID)
    assert result.peaks[0].angle_deg == angle


def test_silence_dc_nyquist_and_independent_noise_return_no_candidate():
    silence = np.zeros((40, len(FREQ), 16), complex)
    silence[:, [0, -1]] = 1
    quiet = subband_candidates(silence, ARRAY, FREQ, GRID)
    assert quiet.peaks == () and quiet.accepted_count == 0
    np.testing.assert_array_equal(quiet.scores, 0)

    rng = np.random.default_rng(19)
    noise = rng.normal(size=silence.shape) + 1j * rng.normal(size=silence.shape)
    result = subband_candidates(noise, ARRAY, FREQ, GRID)
    assert result.peaks == ()


def test_incomplete_frames_and_frequency_tail_are_explicitly_dropped():
    scene = disjoint_scene([15], 47)
    result = subband_candidates(scene, ARRAY, FREQ, GRID, subband_bins=5)
    assert result.frame_count == 40 and result.block_count == 2
    assert result.frequency_bin_count == 45 and result.subband_count == 9
    np.testing.assert_array_equal(
        result.scores,
        subband_candidates(scene[:40], ARRAY, FREQ, GRID, subband_bins=5).scores,
    )


def test_explicit_coordinates_and_reference_index_do_not_change_direction():
    x = (np.arange(16) - 7.5) * 0.03048
    y = np.tile([-0.03, 0.03], 8)
    positions = tuple((float(xx), float(yy), 0.0) for xx, yy in zip(x, y, strict=True))
    array = ArrayConfig(48000, 16, 0.03048, 0, positions_m=positions)
    spectra = np.zeros((20, len(FREQ), 16), complex)
    wave = plane_wave(-22, array)
    active = (FREQ >= 500) & (FREQ <= 5000)
    spectra[:, active] = wave[active]
    result = subband_candidates(spectra, array, FREQ, GRID)
    assert result.peaks[0].angle_deg == -22


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_frames": True},
        {"block_frames": 0},
        {"subband_bins": True},
        {"subband_bins": 0},
        {"max_sources": True},
        {"minimum_bin_score": np.nan},
        {"minimum_bin_prominence": -1},
        {"prominence_radius_deg": 0},
        {"boundary_prominence_radius_deg": 9},
        {"vote_radius_deg": 0},
        {"minimum_support": 1.1},
        {"minimum_separation_deg": -1},
        {"relative_energy_floor": 1},
        {"relative_cross_floor": 1},
        {"min_frequency_hz": 5000},
        {"max_frequency_hz": np.inf},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        subband_candidates(disjoint_scene([0], 20), ARRAY, FREQ, GRID, **kwargs)


def test_invalid_spectral_contracts_and_too_short_valid_input():
    spectra = disjoint_scene([0], 20)
    for bad in [spectra[0], spectra.real, spectra[:, :, :1], np.full_like(spectra, np.nan)]:
        with pytest.raises(ValueError):
            subband_candidates(bad, ARRAY, FREQ, GRID)
    with pytest.raises(ValueError):
        subband_candidates(spectra, ARRAY, FREQ[::-1], GRID)
    with pytest.raises(ValueError):
        subband_candidates(spectra, ARRAY, FREQ, GRID[::-1])
    result = subband_candidates(spectra[:19], ARRAY, FREQ, GRID)
    assert result.frame_count == 0 and result.peaks == ()
