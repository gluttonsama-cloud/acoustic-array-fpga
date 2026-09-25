"""Independent coordinate, source-count, and input checks for the external MUSIC baseline."""

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.localization.music_reference import music_candidates

ARRAY = ArrayConfig(48000, 16, 0.03048, 7)
FREQ = np.fft.rfftfreq(512, 1 / 48000)
GRID = np.arange(-90.0, 91.0)


def test_frequency_mask_excludes_unobserved_covariances() -> None:
    spectra = plane_wave(23, 64, 4)
    keep = (FREQ >= 1500) & (FREQ <= 3500)
    spectra[:, ~keep] = 0
    result = music_candidates(spectra, ARRAY, FREQ, GRID, num_sources=1, frequency_mask=keep)
    assert result.frequency_bin_count == int(keep.sum())
    assert result.peaks[0].angle_deg == pytest.approx(23, abs=1)
    empty = music_candidates(spectra, ARRAY, FREQ, GRID, frequency_mask=np.zeros_like(keep))
    assert empty.peaks == ()
    assert empty.frequency_bin_count == 0
    with pytest.raises(ValueError, match="frequency_mask"):
        music_candidates(spectra, ARRAY, FREQ, GRID, frequency_mask=keep.astype(float))


def plane_wave(angle_deg: float, frames: int, seed: int) -> np.ndarray:
    """Build a far-field wave without calling the project's steering implementation."""
    x = (np.arange(16) - 7.5) * 0.03048
    delay = -x * np.sin(np.deg2rad(angle_deg)) / 343.0
    steering = np.exp(-2j * np.pi * FREQ[:, None] * delay)
    rng = np.random.default_rng(seed)
    snapshots = rng.normal(size=frames) + 1j * rng.normal(size=frames)
    return snapshots[:, None, None] * steering[None, :, :]


@pytest.mark.parametrize("method", ["music", "normmusic"])
@pytest.mark.parametrize("angle_deg", [-35.0, 0.0, 28.0])
def test_single_source_coordinate_conversion(method: str, angle_deg: float) -> None:
    result = music_candidates(
        plane_wave(angle_deg, 64, 1),
        ARRAY,
        FREQ,
        GRID,
        method=method,
        num_sources=1,
        min_frequency_hz=500,
        max_frequency_hz=5000,
    )
    assert len(result.peaks) == 1
    assert result.peaks[0].angle_deg == pytest.approx(angle_deg, abs=1.0)
    assert result.frame_count == 64
    assert result.frequency_bin_count == 48
    assert result.scores.shape == GRID.shape
    assert np.isfinite(result.scores).all()
    assert np.max(result.scores) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["music", "normmusic"])
def test_three_independent_sources_are_not_collapsed_to_one(method: str) -> None:
    spectra = sum(
        plane_wave(angle, 160, seed) for angle, seed in [(-42.0, 11), (0.0, 22), (37.0, 33)]
    )
    result = music_candidates(
        spectra,
        ARRAY,
        FREQ,
        GRID,
        method=method,
        num_sources=3,
        min_frequency_hz=500,
        max_frequency_hz=5000,
        minimum_separation_deg=10,
    )
    assert sorted(peak.angle_deg for peak in result.peaks) == pytest.approx([-42, 0, 37], abs=1)


def test_explicit_planar_geometry_uses_xy_axes() -> None:
    positions = tuple(
        (float(x), float(y), 0.0)
        for x, y in [(-0.06, -0.02), (-0.02, 0.02), (0.02, -0.02), (0.06, 0.02)]
    )
    array = ArrayConfig(48000, 4, 0.04, 1, positions_m=positions)
    angle = -24.0
    direction = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))])
    delays = -(np.asarray(positions)[:, :2] @ direction) / 343.0
    steering = np.exp(-2j * np.pi * FREQ[:, None] * delays)
    rng = np.random.default_rng(7)
    spectra = (rng.normal(size=80) + 1j * rng.normal(size=80))[:, None, None] * steering[None]
    result = music_candidates(spectra, array, FREQ, GRID, method="normmusic", num_sources=1)
    assert result.peaks[0].angle_deg == pytest.approx(angle, abs=1)


@pytest.mark.parametrize("method", ["music", "normmusic"])
def test_silence_returns_no_candidates(method: str) -> None:
    result = music_candidates(
        np.zeros((64, len(FREQ), 16), dtype=complex),
        ARRAY,
        FREQ,
        GRID,
        method=method,
    )
    np.testing.assert_array_equal(result.scores, 0)
    assert result.peaks == ()


@pytest.mark.parametrize("method", ["music", "normmusic"])
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_extreme_common_gain_preserves_direction(method: str, scale: float) -> None:
    result = music_candidates(
        plane_wave(23, 64, 4) * scale,
        ARRAY,
        FREQ,
        GRID,
        method=method,
        num_sources=1,
    )
    assert result.peaks[0].angle_deg == pytest.approx(23, abs=1)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda x: x[0], "spectra"),
        (lambda x: x.copy().astype(complex), "finite"),
    ],
)
def test_invalid_spectrum_contracts(change, match: str) -> None:
    spectra = plane_wave(0, 64, 1)
    changed = change(spectra)
    if match == "finite":
        changed[0, 2, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        music_candidates(changed, ARRAY, FREQ, GRID)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "srp"},
        {"num_sources": 0},
        {"num_sources": True},
        {"num_sources": 16},
        {"angles_deg": GRID[::-1]},
        {"min_frequency_hz": 6000, "max_frequency_hz": 5000},
        {"minimum_separation_deg": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    arguments = {
        "spectra": plane_wave(0, 64, 1),
        "array": ARRAY,
        "frequencies_hz": FREQ,
        "angles_deg": GRID,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        music_candidates(**arguments)
