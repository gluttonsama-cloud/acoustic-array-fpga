"""Bounded temporal SRP-PHAT candidate voting for front-half-plane directions.

The input axes are [frame, frequency, microphone]. Only complete, nonoverlapping
blocks are used. A score is the fraction of blocks voting near an angle, not a
probability of source presence or a calibrated detection confidence.
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


@dataclass(frozen=True)
class TemporalResult:
    scores: NDArray[np.float64]
    block_scores: NDArray[np.float64]
    peaks: tuple[DirectionPeak, ...]
    block_count: int
    frame_count: int


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _prominent_block_peaks(
    scores: NDArray[np.float64],
    angles: NDArray[np.float64],
    minimum_score: float,
    minimum_prominence: float,
    prominence_radius_deg: float,
    boundary_prominence_radius_deg: float,
    minimum_separation_deg: float,
    max_sources: int,
) -> tuple[DirectionPeak, ...]:
    # First enumerate all local maxima, then apply prominence before NMS/top-K.
    local = select_peaks(
        scores,
        angles,
        minimum_score=minimum_score,
        minimum_separation_deg=0.0,
        max_sources=angles.size,
    )
    eligible: list[DirectionPeak] = []
    for peak in local:
        delta = angles - peak.angle_deg
        radius = (
            boundary_prominence_radius_deg
            if peak.angle_deg == angles[0] or peak.angle_deg == angles[-1]
            else prominence_radius_deg
        )
        left = scores[(delta < 0) & (delta >= -radius)]
        right = scores[(delta > 0) & (delta <= radius)]
        flank_minima = [float(side.min()) for side in (left, right) if side.size]
        if flank_minima and peak.score - max(flank_minima) >= minimum_prominence:
            eligible.append(peak)
    if not eligible:
        return ()
    # Preserve select_peaks' local-maximum and greedy-suppression semantics.
    filtered = np.zeros_like(scores)
    for peak in eligible:
        index = int(np.searchsorted(angles, peak.angle_deg))
        filtered[index] = peak.score
    return select_peaks(
        filtered,
        angles,
        minimum_score=minimum_score,
        minimum_separation_deg=minimum_separation_deg,
        max_sources=max_sources,
    )


def temporal_candidates(
    spectra: ArrayLike,
    array: ArrayConfig,
    frequencies_hz: ArrayLike,
    angles_deg: ArrayLike,
    *,
    block_frames: int = 20,
    min_frequency_hz: float = 500.0,
    max_frequency_hz: float = 5000.0,
    relative_floor: float = 1e-8,
    minimum_score: float = 0.1,
    minimum_prominence: float = 0.05,
    prominence_radius_deg: float = 10.0,
    boundary_prominence_radius_deg: float = 30.0,
    vote_radius_deg: float = 3.0,
    minimum_support: float = 0.1,
    minimum_separation_deg: float = 10.0,
    max_sources: int = 3,
) -> TemporalResult:
    """Return per-block maps and a bounded angular vote from complete blocks.

    Each active time-frequency-pair observation is phase-normalized before its
    block average. A frame's cross-power floor is relative to that frame's
    largest eligible pair/frequency magnitude. Silence contributes zero.
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
    source_limit = _positive_integer(max_sources, "max_sources")
    low = _real_scalar(min_frequency_hz, "min_frequency_hz")
    high = _real_scalar(max_frequency_hz, "max_frequency_hz")
    floor = _real_scalar(relative_floor, "relative_floor")
    block_threshold = _real_scalar(minimum_score, "minimum_score")
    prominence = _real_scalar(minimum_prominence, "minimum_prominence")
    prominence_radius = _real_scalar(prominence_radius_deg, "prominence_radius_deg")
    boundary_radius = _real_scalar(boundary_prominence_radius_deg, "boundary_prominence_radius_deg")
    vote_radius = _real_scalar(vote_radius_deg, "vote_radius_deg")
    support = _real_scalar(minimum_support, "minimum_support")
    separation = _real_scalar(minimum_separation_deg, "minimum_separation_deg")
    if not 0 <= low < high <= nyquist:
        raise ValueError("frequency band must satisfy 0 <= min < max <= Nyquist")
    if not 0 <= floor < 1:
        raise ValueError("relative_floor must lie in [0, 1)")
    if not 0 <= block_threshold <= 1 or not 0 <= prominence <= 2:
        raise ValueError("minimum_score and minimum_prominence must lie in [0,1] and [0,2]")
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
    block_scores = np.zeros((block_count, angles.size), dtype=np.float64)
    votes = np.zeros(angles.size, dtype=np.float64)
    eligible = (frequencies >= low) & (frequencies <= high)
    eligible &= (frequencies > 0) & (frequencies < nyquist)
    if block_count == 0 or not np.any(eligible):
        return TemporalResult(votes, block_scores, (), block_count, frame_count)

    frequencies = frequencies[eligible]
    left, right = np.triu_indices(array.microphone_count, k=1)
    steering = steering_vectors(array, frequencies, angles)
    pair_model = steering[:, left, :] * steering[:, right, :].conj()
    model_conjugate = pair_model.conj()
    for block in range(block_count):
        part = x[block * block_size : (block + 1) * block_size, eligible, :]
        # Scale each frame separately before products, avoiding overflow and
        # preserving phase for extremely small or large coherent frames.
        scale = np.maximum(
            np.max(np.abs(part.real), axis=(1, 2)), np.max(np.abs(part.imag), axis=(1, 2))
        )
        safe_scale = np.where(scale > 0, scale, 1.0)
        normalized = part.astype(np.complex128, copy=False) / safe_scale[:, None, None]
        cross = normalized[:, :, left] * normalized[:, :, right].conj()
        magnitude = np.abs(cross)
        frame_max = np.max(magnitude, axis=(1, 2))
        valid = (magnitude > 0) & (magnitude >= floor * frame_max[:, None, None])
        phat = np.zeros_like(cross)
        np.divide(cross, magnitude, out=phat, where=valid)
        mean_phat = np.sum(phat, axis=0) / block_size
        block_scores[block] = np.real(
            np.einsum(
                "fp,fpa->a",
                mean_phat,
                model_conjugate,
                optimize=True,
            )
        ) / (frequencies.size * left.size)
        peaks = _prominent_block_peaks(
            block_scores[block],
            angles,
            block_threshold,
            prominence,
            prominence_radius,
            boundary_radius,
            separation,
            source_limit,
        )
        if peaks:
            block_vote = np.maximum.reduce(
                [
                    np.maximum(0.0, 1.0 - np.abs(angles - peak.angle_deg) / vote_radius)
                    for peak in peaks
                ]
            )
            votes += block_vote / block_count

    final_peaks = select_peaks(
        votes,
        angles,
        minimum_score=support,
        minimum_separation_deg=separation,
        max_sources=source_limit,
    )
    return TemporalResult(votes, block_scores, final_peaks, block_count, frame_count)
