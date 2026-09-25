"""Time-frequency subband candidates from bounded SRP-PHAT votes.

Spectra have axes [frame, frequency, microphone]. Only complete time blocks
and complete contiguous eligible-frequency subbands contribute to the result.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig
from acoustic_array.core.geometry import steering_vectors
from acoustic_array.localization.scan import (
    DirectionPeak,
    _angles,
    _real_scalar,
    _real_vector,
    select_peaks,
)
from acoustic_array.localization.temporal import _positive_integer


@dataclass(frozen=True)
class SubbandResult:
    scores: NDArray[np.float64]
    subband_scores: NDArray[np.float64]
    peaks: tuple[DirectionPeak, ...]
    block_count: int
    frame_count: int
    subband_count: int
    frequency_bin_count: int
    accepted_count: int


def _strongest_prominent_peak(
    scores: NDArray[np.float64],
    angles: NDArray[np.float64],
    minimum_score: float,
    minimum_prominence: float,
    prominence_radius_deg: float,
    boundary_prominence_radius_deg: float,
) -> DirectionPeak | None:
    maxima = select_peaks(
        scores,
        angles,
        minimum_score=minimum_score,
        minimum_separation_deg=0.0,
        max_sources=angles.size,
    )
    if not maxima:
        return None
    peak = maxima[0]
    delta = angles - peak.angle_deg
    radius = (
        boundary_prominence_radius_deg
        if peak.angle_deg == angles[0] or peak.angle_deg == angles[-1]
        else prominence_radius_deg
    )
    left = scores[(delta < 0) & (delta >= -radius)]
    right = scores[(delta > 0) & (delta <= radius)]
    minima = [float(side.min()) for side in (left, right) if side.size]
    if not minima or peak.score - max(minima) < minimum_prominence:
        return None
    return peak


def subband_candidates(
    spectra: ArrayLike,
    array: ArrayConfig,
    frequencies_hz: ArrayLike,
    angles_deg: ArrayLike,
    *,
    block_frames: int = 20,
    subband_bins: int = 4,
    min_frequency_hz: float = 500.0,
    max_frequency_hz: float = 5000.0,
    relative_energy_floor: float = 1e-5,
    relative_cross_floor: float = 1e-8,
    minimum_bin_score: float = 0.2,
    minimum_bin_prominence: float = 0.05,
    prominence_radius_deg: float = 10.0,
    boundary_prominence_radius_deg: float = 30.0,
    vote_radius_deg: float = 3.0,
    minimum_support: float = 0.05,
    minimum_separation_deg: float = 10.0,
    max_sources: int = 3,
) -> SubbandResult:
    """Return a normalized angular vote and one candidate per active subband slot.

    A subband scan uses per-frame PHAT before averaging and retains the fixed
    frame, frequency-bin and microphone-pair denominator. Inactive or rejected
    slots therefore contribute zero, including in the final vote denominator.
    """
    if not isinstance(array, ArrayConfig) or array.microphone_count < 2:
        raise ValueError("array must have at least two microphones")
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

    block_size = _positive_integer(block_frames, "block_frames")
    band_size = _positive_integer(subband_bins, "subband_bins")
    source_limit = _positive_integer(max_sources, "max_sources")
    low = _real_scalar(min_frequency_hz, "min_frequency_hz")
    high = _real_scalar(max_frequency_hz, "max_frequency_hz")
    energy_floor = _real_scalar(relative_energy_floor, "relative_energy_floor")
    cross_floor = _real_scalar(relative_cross_floor, "relative_cross_floor")
    bin_threshold = _real_scalar(minimum_bin_score, "minimum_bin_score")
    prominence = _real_scalar(minimum_bin_prominence, "minimum_bin_prominence")
    prominence_radius = _real_scalar(prominence_radius_deg, "prominence_radius_deg")
    boundary_radius = _real_scalar(boundary_prominence_radius_deg, "boundary_prominence_radius_deg")
    vote_radius = _real_scalar(vote_radius_deg, "vote_radius_deg")
    support = _real_scalar(minimum_support, "minimum_support")
    separation = _real_scalar(minimum_separation_deg, "minimum_separation_deg")
    if not 0 <= low < high <= nyquist:
        raise ValueError("frequency band must satisfy 0 <= min < max <= Nyquist")
    if not 0 <= energy_floor < 1 or not 0 <= cross_floor < 1:
        raise ValueError("relative floors must lie in [0, 1)")
    if not 0 <= bin_threshold <= 1 or not 0 <= prominence <= 2:
        raise ValueError("minimum score and prominence must lie in [0,1] and [0,2]")
    if (
        prominence_radius <= 0
        or boundary_radius < prominence_radius
        or vote_radius <= 0
        or not 0 <= support <= 1
        or separation < 0
    ):
        raise ValueError(
            "radii must be positive, boundary radius >= interior radius, "
            "support in [0,1], separation nonnegative"
        )

    block_count = x.shape[0] // block_size
    frame_count = block_count * block_size
    eligible = (frequencies >= low) & (frequencies <= high)
    eligible &= (frequencies > 0) & (frequencies < nyquist)
    eligible_indices = np.flatnonzero(eligible)
    subband_count = eligible_indices.size // band_size
    frequency_bin_count = subband_count * band_size
    indices = eligible_indices[:frequency_bin_count]
    maps = np.zeros((block_count, subband_count, angles.size), dtype=np.float64)
    votes = np.zeros(angles.size, dtype=np.float64)
    if block_count == 0 or subband_count == 0:
        return SubbandResult(
            votes, maps, (), block_count, frame_count, subband_count, frequency_bin_count, 0
        )

    left, right = np.triu_indices(array.microphone_count, k=1)
    accepted_count = 0
    for block in range(block_count):
        part = x[block * block_size : (block + 1) * block_size, indices, :]
        # Scaling each frame by a component maximum keeps both extreme common
        # gains and silent frames finite before the cross products are formed.
        scale = np.maximum(
            np.max(np.abs(part.real), axis=(1, 2)),
            np.max(np.abs(part.imag), axis=(1, 2)),
        )
        normalized = (
            part.astype(np.complex128, copy=False) / np.where(scale > 0, scale, 1.0)[:, None, None]
        )
        bands = normalized.reshape(block_size, subband_count, band_size, array.microphone_count)
        # One scale for the entire block preserves power ratios across frames.
        # Per-frame scaling above is used only for the phase-based PHAT scan.
        block_scale = float(np.max(scale))
        energy_bands = (part / (block_scale if block_scale > 0 else 1.0)).reshape(
            block_size, subband_count, band_size, array.microphone_count
        )
        energy = np.mean(np.abs(energy_bands) ** 2, axis=(0, 2, 3))
        active = (energy > 0) & (energy >= energy_floor * np.max(energy))
        for band in np.flatnonzero(active):
            band_x = bands[:, band]
            cross = band_x[:, :, left] * band_x[:, :, right].conj()
            magnitude = np.abs(cross)
            frame_max = np.max(magnitude, axis=(1, 2))
            valid = (magnitude > 0) & (magnitude >= cross_floor * frame_max[:, None, None])
            phat = np.zeros_like(cross)
            np.divide(cross, magnitude, out=phat, where=valid)
            mean_phat = np.sum(phat, axis=0) / block_size
            band_frequencies = frequencies[indices[band * band_size : (band + 1) * band_size]]
            steering = steering_vectors(array, band_frequencies, angles)
            model = steering[:, left, :] * steering[:, right, :].conj()
            scores = np.real(np.einsum("fp,fpa->a", mean_phat, model.conj(), optimize=True)) / (
                band_size * left.size
            )
            maps[block, band] = scores
            peak = _strongest_prominent_peak(
                scores,
                angles,
                bin_threshold,
                prominence,
                prominence_radius,
                boundary_radius,
            )
            if peak is not None:
                accepted_count += 1
                votes += np.maximum(0.0, 1.0 - np.abs(angles - peak.angle_deg) / vote_radius)

    votes /= block_count * subband_count
    peaks = select_peaks(
        votes,
        angles,
        minimum_score=support,
        minimum_separation_deg=separation,
        max_sources=source_limit,
    )
    return SubbandResult(
        votes,
        maps,
        peaks,
        block_count,
        frame_count,
        subband_count,
        frequency_bin_count,
        accepted_count,
    )
