"""Offline MUSIC baselines with the project's front-half-plane angle convention.

This module is an adapter for pyroomacoustics, not a deployable FPGA algorithm.
Input spectra use [frame, frequency, microphone] axes and a complete real-FFT
frequency grid. The library receives [microphone, frequency, snapshot].
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig
from acoustic_array.core.geometry import microphone_positions
from acoustic_array.localization.scan import (
    DirectionPeak,
    _angles,
    _real_scalar,
    _real_vector,
    select_peaks,
)


@dataclass(frozen=True)
class MusicResult:
    """A finite, unit-maximum scan and selected candidate directions."""

    scores: NDArray[np.float64]
    peaks: tuple[DirectionPeak, ...]
    frame_count: int
    frequency_bin_count: int
    method: str
    num_sources: int


def music_candidates(
    spectra: ArrayLike,
    array: ArrayConfig,
    frequencies_hz: ArrayLike,
    angles_deg: ArrayLike,
    *,
    method: str = "normmusic",
    num_sources: int = 3,
    min_frequency_hz: float = 500.0,
    max_frequency_hz: float = 5000.0,
    minimum_separation_deg: float = 10.0,
    frequency_mask: ArrayLike | None = None,
) -> MusicResult:
    """Evaluate MUSIC or NormMUSIC on an explicit half-plane angular grid.

    ``num_sources`` specifies the signal-subspace rank. Peak selection then
    applies the project's independent local-maximum and angular-spacing rule.
    Scores are relative scan strengths, not probabilities.
    """
    if not isinstance(array, ArrayConfig):
        raise ValueError("array must be an ArrayConfig")
    x = np.asarray(spectra)
    if x.ndim != 3 or any(size == 0 for size in x.shape) or not np.iscomplexobj(x):
        raise ValueError("spectra must be a nonempty complex [frame,frequency,mic] array")
    if x.shape[2] != array.microphone_count or not np.isfinite(x).all():
        raise ValueError("spectra microphone count must match array and values must be finite")
    frequencies = _real_vector(frequencies_hz, "frequencies_hz")
    angles = _angles(angles_deg)
    if frequencies.size != x.shape[1] or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies_hz must match spectra and increase strictly")
    nyquist = array.sample_rate_hz / 2
    if frequencies[0] < 0 or frequencies[-1] > nyquist:
        raise ValueError("frequencies_hz must lie in [0, Nyquist]")
    if method not in ("music", "normmusic"):
        raise ValueError("method must be 'music' or 'normmusic'")
    if (
        isinstance(num_sources, (bool, np.bool_))
        or not isinstance(num_sources, (int, np.integer))
        or not 1 <= num_sources < array.microphone_count
    ):
        raise ValueError("num_sources must be a positive integer below microphone_count")
    low = _real_scalar(min_frequency_hz, "min_frequency_hz")
    high = _real_scalar(max_frequency_hz, "max_frequency_hz")
    separation = _real_scalar(minimum_separation_deg, "minimum_separation_deg")
    if not 0 <= low < high <= nyquist:
        raise ValueError("frequency band must satisfy 0 <= min < max <= Nyquist")
    if separation <= 0:
        raise ValueError("minimum_separation_deg must be positive")

    # pyroomacoustics' mode vectors index an RFFT with a known nfft. Accepting
    # arbitrary frequencies would silently steer at the wrong physical frequency.
    nfft = 2 * (frequencies.size - 1)
    expected = np.fft.rfftfreq(nfft, 1 / array.sample_rate_hz) if nfft > 0 else []
    if nfft < 4 or not np.allclose(frequencies, expected, rtol=0, atol=1e-7):
        raise ValueError("frequencies_hz must be the complete RFFT grid for the sample rate")
    bins = np.flatnonzero(
        (frequencies >= low) & (frequencies <= high) & (frequencies > 0) & (frequencies < nyquist)
    )
    if bins.size == 0:
        raise ValueError("frequency band contains no eligible non-DC, non-Nyquist bins")
    if frequency_mask is not None:
        mask = np.asarray(frequency_mask)
        if mask.dtype != np.bool_ or mask.shape != frequencies.shape:
            raise ValueError("frequency_mask must be a boolean vector matching frequencies")
        bins = bins[mask[bins]]
        if bins.size == 0:
            return MusicResult(np.zeros_like(angles), (), x.shape[0], 0, method, int(num_sources))
    scale = float(np.max(np.abs(x[:, bins, :])))
    if scale == 0:
        return MusicResult(
            np.zeros_like(angles), (), x.shape[0], bins.size, method, int(num_sources)
        )

    # MUSIC depends on covariance subspaces rather than common gain. Normalize
    # only the selected band and zero unused bins to avoid under/overflow in the
    # external covariance calculation without inventing an absolute level gate.
    scaled = np.zeros(x.shape, dtype=np.complex128)
    scaled[:, bins, :] = x[:, bins, :] / scale

    try:
        import pyroomacoustics as pra
    except ImportError as exc:
        raise ImportError(
            "MUSIC reference requires pyroomacoustics; install the project's 'doa' extra"
        ) from exc

    # Project theta is measured from +y toward +x; library phi is from +x.
    # GridCircle sorts phi internally, so map scores back by its actual azimuth.
    phi = np.deg2rad(90.0 - angles)
    positions = microphone_positions(array)[:, :2].T
    estimator_type = pra.doa.MUSIC if method == "music" else pra.doa.NormMUSIC
    estimator = estimator_type(
        positions,
        fs=array.sample_rate_hz,
        nfft=nfft,
        c=array.sound_speed_m_per_s,
        num_src=int(num_sources),
        mode="far",
        azimuth=phi,
    )
    library_spectra = np.transpose(scaled, (2, 1, 0))
    estimator.locate_sources(library_spectra, freq_bins=bins)
    library_scores = np.asarray(estimator.grid.values, dtype=np.float64)
    library_angles = 90.0 - np.rad2deg(estimator.grid.azimuth)
    if library_scores.shape != angles.shape or not np.isfinite(library_scores).all():
        raise ValueError("MUSIC produced nonfinite or unexpected scan scores")
    order = np.argsort(library_angles)
    if not np.allclose(library_angles[order], angles, rtol=0, atol=1e-8):
        raise ValueError("MUSIC returned an unexpected angular grid")
    scores = library_scores[order]
    maximum = float(np.max(scores))
    if maximum <= 0:
        normalized = np.zeros_like(scores)
        peaks: tuple[DirectionPeak, ...] = ()
    else:
        normalized = scores / maximum
        normalized = np.clip(normalized, 0.0, 1.0)
        peaks = select_peaks(
            normalized,
            angles,
            minimum_score=0.0,
            minimum_separation_deg=separation,
            max_sources=int(num_sources),
        )
    return MusicResult(normalized, peaks, x.shape[0], bins.size, method, int(num_sources))
