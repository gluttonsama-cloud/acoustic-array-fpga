"""Static, far-field direction scanning and independent peak selection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig
from acoustic_array.core.geometry import steering_vectors


@dataclass(frozen=True)
class DirectionPeak:
    angle_deg: float
    score: float


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty real vector")
    result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _real_scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real number")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _angles(values: ArrayLike) -> NDArray[np.float64]:
    angles = _real_vector(values, "angles_deg")
    if np.any(np.abs(angles) > 90) or np.any(np.diff(angles) <= 0):
        raise ValueError("angles_deg must be strictly increasing within [-90, 90]")
    return angles


def scan_directions(
    spectra: ArrayLike,
    array: ArrayConfig,
    frequencies_hz: ArrayLike,
    angles_deg: ArrayLike,
    *,
    method: str = "srp_phat",
    min_frequency_hz: float = 500.0,
    max_frequency_hz: float = 5000.0,
    relative_floor: float = 1e-8,
) -> NDArray[np.float64]:
    """Return one static far-field scan score per front-half-plane angle.

    Input axes are [frame, frequency, microphone]. Both methods average over
    eligible frequencies. ``das`` returns mean steered output power after
    scaling all input spectra by their maximum component magnitude. ``srp_phat`` first
    averages each microphone-pair cross spectrum over frames, then uses its
    phase for a real, off-diagonal pair average. Its score is not a probability.
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
    if method not in ("das", "srp_phat"):
        raise ValueError("method must be 'das' or 'srp_phat'")
    low = _real_scalar(min_frequency_hz, "min_frequency_hz")
    high = _real_scalar(max_frequency_hz, "max_frequency_hz")
    floor = _real_scalar(relative_floor, "relative_floor")
    if not 0 <= low < high <= nyquist:
        raise ValueError("frequency band must satisfy 0 <= min < max <= Nyquist")
    if not 0 <= floor < 1:
        raise ValueError("relative_floor must lie in [0, 1)")
    if method == "srp_phat" and array.microphone_count < 2:
        raise ValueError("srp_phat needs at least two microphones")

    eligible = (frequencies >= low) & (frequencies <= high)
    eligible &= (frequencies > 0) & (frequencies < nyquist)
    if not np.any(eligible):
        return np.zeros(angles.size, dtype=np.float64)
    x = x[:, eligible, :].astype(np.complex128, copy=False)
    frequencies = frequencies[eligible]
    # Componentwise scaling avoids overflow in abs() for large finite complex values.
    scale = float(max(np.max(np.abs(x.real)), np.max(np.abs(x.imag))))
    if scale == 0:
        return np.zeros(angles.size, dtype=np.float64)
    x = x / scale
    energy = np.mean(np.abs(x) ** 2, axis=(0, 2))
    active = (energy > 0) & (energy >= floor * np.max(energy))
    if not np.any(active):
        return np.zeros(angles.size, dtype=np.float64)
    x = x[:, active, :]
    frequencies = frequencies[active]
    steering = steering_vectors(array, frequencies, angles)

    if method == "das":
        steered = np.einsum("tfm,fma->tfa", x, steering.conj(), optimize=True)
        return np.mean(np.abs(steered / array.microphone_count) ** 2, axis=(0, 1))

    left, right = np.triu_indices(array.microphone_count, k=1)
    cross = np.mean(x[:, :, left] * x[:, :, right].conj(), axis=0)
    strength = np.abs(cross)
    valid = (strength > 0) & (strength >= floor * np.max(strength))
    if not np.any(valid):
        return np.zeros(angles.size, dtype=np.float64)
    phat = np.zeros_like(cross)
    phat[valid] = cross[valid] / strength[valid]
    model = steering[:, left, :] * steering[:, right, :].conj()
    aligned = np.real(phat[:, :, None] * model.conj())
    return np.sum(aligned * valid[:, :, None], axis=(0, 1)) / np.count_nonzero(valid)


def select_peaks(
    scores: ArrayLike,
    angles_deg: ArrayLike,
    *,
    minimum_score: float,
    minimum_separation_deg: float = 10.0,
    max_sources: int = 3,
) -> tuple[DirectionPeak, ...]:
    """Choose local maxima, strongest first, with greedy angular suppression.

    A flat summit contributes its center grid point (lower middle point for
    even lengths). A wholly flat scan has no informative local maximum.
    """
    values = _real_vector(scores, "scores")
    angles = _angles(angles_deg)
    if values.size != angles.size:
        raise ValueError("scores and angles_deg must have the same length")
    threshold = _real_scalar(minimum_score, "minimum_score")
    separation = _real_scalar(minimum_separation_deg, "minimum_separation_deg")
    if separation < 0:
        raise ValueError("minimum_separation_deg must be nonnegative")
    if (
        isinstance(max_sources, (bool, np.bool_))
        or not isinstance(max_sources, (int, np.integer))
        or max_sources <= 0
    ):
        raise ValueError("max_sources must be a positive integer")
    if not np.any(values != 0):
        return ()

    candidates: list[int] = []
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and values[end + 1] == values[start]:
            end += 1
        if values.size == 1 or start > 0 or end + 1 < values.size:
            left_lower = start == 0 or values[start] > values[start - 1]
            right_lower = end + 1 == values.size or values[end] > values[end + 1]
            if left_lower and right_lower and values[start] >= threshold:
                candidates.append((start + end) // 2)
        start = end + 1

    candidates.sort(key=lambda i: (-values[i], angles[i]))
    selected: list[DirectionPeak] = []
    for index in candidates:
        if all(abs(angles[index] - peak.angle_deg) >= separation for peak in selected):
            selected.append(DirectionPeak(float(angles[index]), float(values[index])))
            if len(selected) == max_sources:
                break
    return tuple(selected)
