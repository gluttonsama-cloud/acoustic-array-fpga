"""Independent plane-wave, peak and PCM chunking checks for static localization."""

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.dsp.stft import StreamingSTFT
from acoustic_array.localization.scan import scan_directions, select_peaks


def plane_wave(angle, frequencies, frames=8):
    # Analytic ULA travel path, deliberately independent of steering_vectors.
    x = (np.arange(16) - 7.5) * 0.03048
    delay = -x * np.sin(np.deg2rad(angle)) / 343
    phase = np.exp(-2j * np.pi * frequencies[:, None] * delay)
    return np.broadcast_to(phase, (frames, *phase.shape)).copy()


ARRAY = ArrayConfig(48000, 16, 0.03048, 7)
FREQ = np.fft.rfftfreq(512, 1 / 48000)
GRID = np.arange(-90.0, 91.0)


@pytest.mark.parametrize("method", ["das", "srp_phat"])
@pytest.mark.parametrize("angle", [-35.0, 0.0, 28.0])
def test_single_source_sign_and_reference_invariance(method, angle):
    spectra = plane_wave(angle, FREQ)
    score = scan_directions(spectra, ARRAY, FREQ, GRID, method=method)
    assert GRID[np.argmax(score)] == angle
    other = ArrayConfig(48000, 16, 0.03048, 0)
    np.testing.assert_allclose(
        score, scan_directions(spectra, other, FREQ, GRID, method=method), atol=1e-12
    )


@pytest.mark.parametrize("method", ["das", "srp_phat"])
def test_silence_and_nyquist_do_not_create_sources(method):
    spectra = np.zeros((4, len(FREQ), 16), complex)
    spectra[:, [0, -1]] = 1
    scores = scan_directions(spectra, ARRAY, FREQ, GRID, method=method)
    np.testing.assert_array_equal(scores, 0)
    assert select_peaks(scores, GRID, minimum_score=0.01) == ()


def test_two_sources_with_orthogonal_temporal_codes():
    left, right = plane_wave(-30, FREQ), plane_wave(30, FREQ)
    right[1::2] *= -1  # Exactly zero temporal cross-correlation.
    scores = scan_directions(left + right, ARRAY, FREQ, GRID)
    peaks = select_peaks(scores, GRID, minimum_score=0.2, max_sources=2)
    assert len(peaks) == 2
    assert sorted(p.angle_deg for p in peaks) == pytest.approx([-30, 30], abs=1)


def test_peak_suppression_threshold_plateau_and_empty():
    angles = np.arange(-4.0, 5.0)
    scores = [0, 0.8, 0, 0.9, 0.9, 0, 0.7, 0, 0.1]
    peaks = select_peaks(scores, angles, minimum_score=0.5, minimum_separation_deg=3)
    assert len(peaks) == 2
    assert peaks[0].score == 0.9
    assert abs(peaks[0].angle_deg - peaks[1].angle_deg) >= 3
    assert select_peaks(scores, angles, minimum_score=1.0) == ()


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_common_gain_preserves_direction(scale):
    spectra = plane_wave(23, FREQ)
    for method in ["das", "srp_phat"]:
        score = scan_directions(spectra * scale, ARRAY, FREQ, GRID, method=method)
        assert np.isfinite(score).all()
        assert GRID[np.argmax(score)] == 23


def test_pcm_block_boundaries_preserve_spatial_spectrum():
    pcm = np.random.default_rng(4).normal(size=(2101, 16))

    def analyze(block):
        stft = StreamingSTFT(STFTConfig(512, 256), 16)
        frames = []
        for start in range(0, len(pcm), block):
            frames.extend(stft.push(pcm[start : start + block], sample_start=start))
        frames.extend(stft.flush())
        return np.stack([f.spectrum for f in frames])

    one, split = analyze(len(pcm)), analyze(137)
    np.testing.assert_array_equal(one, split)
    np.testing.assert_array_equal(
        scan_directions(one, ARRAY, FREQ, GRID), scan_directions(split, ARRAY, FREQ, GRID)
    )


def test_invalid_input_contracts():
    spectra = plane_wave(0, FREQ)
    with pytest.raises(ValueError):
        scan_directions(spectra[0], ARRAY, FREQ, GRID)
    with pytest.raises(ValueError):
        scan_directions(spectra, ARRAY, FREQ[::-1], GRID)
    with pytest.raises(ValueError):
        scan_directions(spectra, ARRAY, FREQ, GRID[::-1])
    spectra[0, 2, 0] = np.nan
    with pytest.raises(ValueError):
        scan_directions(spectra, ARRAY, FREQ, GRID)
    with pytest.raises(ValueError):
        select_peaks([0, 1, 0], [-1, 0, 1], minimum_score=0.1, max_sources=True)


def test_explicit_planar_geometry_and_common_phase():
    positions = ((-0.04, -0.03, 0), (0.04, -0.03, 0), (-0.04, 0.03, 0), (0.04, 0.03, 0))
    array = ArrayConfig(48000, 4, 0.08, 0, positions_m=positions)
    angle = np.deg2rad(-22)
    paths = [-x * np.sin(angle) - y * np.cos(angle) for x, y, _ in positions]
    signal = np.exp(-2j * np.pi * FREQ[:, None] * np.asarray(paths) / 343)[None]
    for method in ["das", "srp_phat"]:
        scores = scan_directions(signal, array, FREQ, GRID, method=method)
        shifted = scan_directions(signal * 1j, array, FREQ, GRID, method=method)
        assert GRID[np.argmax(scores)] == -22
        np.testing.assert_allclose(scores, shifted, atol=1e-12)


def test_uncorrelated_noise_is_not_forced_to_three_sources():
    rng = np.random.default_rng(42)
    shape = (128, len(FREQ), 16)
    spectra = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    scores = scan_directions(spectra, ARRAY, FREQ, GRID)
    assert select_peaks(scores, GRID, minimum_score=0.1) == ()
