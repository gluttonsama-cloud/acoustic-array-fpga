"""Batch time-frequency onset selection without source labels or file IO.

This is a simplified candidate filter inspired by the log cross-spectrum rise
in arXiv:1904.06648, section 2.1. Top-K uses the entire observation window,
so the result is not causal or an online detector. Transients are not excluded.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class OnsetResult:
    filtered_spectra: NDArray[np.complex128]
    mask: NDArray[np.bool_]
    rise_db: NDArray[np.float64]
    selected_counts: NDArray[np.int64]


def _positive_integer(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_, str, bytes, complex, np.complexfloating)):
        raise ValueError(f"{name} must be a finite real number")
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def onset_filter(
    spectra: ArrayLike,
    *,
    history_frames: int = 4,
    rise_db: float = 3.0,
    keep_fraction: float = 0.2,
    relative_floor: float = 1e-4,
    minimum_snapshots: int = 16,
) -> OnsetResult:
    """Keep strong rising time-frequency bins in a complete batch window.

    The input axes are [frame, frequency, microphone]. Power is the square of
    mean microphone magnitude, equivalent to the mean magnitude of all ordered
    microphone cross-spectra. A frequency is cleared unless at least
    ``minimum_snapshots`` candidates survive its top-K cap.
    """
    x = np.asarray(spectra)
    if x.ndim != 3 or any(size == 0 for size in x.shape) or not np.iscomplexobj(x):
        raise ValueError("spectra must be a nonempty complex [frame,frequency,mic] array")
    if not np.isfinite(x).all():
        raise ValueError("spectra values must be finite")

    history = _positive_integer(history_frames, "history_frames")
    minimum = _positive_integer(minimum_snapshots, "minimum_snapshots")
    threshold = _finite_real(rise_db, "rise_db")
    fraction = _finite_real(keep_fraction, "keep_fraction")
    floor = _finite_real(relative_floor, "relative_floor")
    if threshold <= 0:
        raise ValueError("rise_db must be positive")
    if not 0 < fraction <= 1:
        raise ValueError("keep_fraction must lie in (0, 1]")
    if not 0 <= floor <= 1:
        raise ValueError("relative_floor must lie in [0, 1]")

    frames, frequencies, _ = x.shape
    mask = np.zeros((frames, frequencies), dtype=bool)
    rises = np.zeros((frames, frequencies), dtype=np.float64)
    counts = np.zeros(frequencies, dtype=np.int64)
    if frames <= history:
        return OnsetResult(np.zeros_like(x), mask, rises, counts)

    # Each frequency has its own common microphone/time scale. This protects
    # powers without letting an unrelated strong band set a weak band's log floor.
    scale = np.maximum(np.max(np.abs(x.real), axis=(0, 2)), np.max(np.abs(x.imag), axis=(0, 2)))
    safe_scale = np.where(scale > 0, scale, 1.0)
    normalized = np.empty(x.shape, dtype=np.complex128)
    normalized.real = x.real / safe_scale[None, :, None]
    normalized.imag = x.imag / safe_scale[None, :, None]
    power = np.mean(np.abs(normalized), axis=2) ** 2
    log_power = 10.0 * np.log10(power + 1e-12)
    previous = np.lib.stride_tricks.sliding_window_view(log_power, history, axis=0)[:-1]
    rises[history:] = log_power[history:] - np.mean(previous, axis=-1)

    frequency_max = np.max(power, axis=0)
    eligible = (power > 0) & (power >= floor * frequency_max[None, :])
    cap = int(np.ceil((frames - history) * fraction))
    for frequency in range(frequencies):
        candidates = np.flatnonzero(
            eligible[history:, frequency] & (rises[history:, frequency] >= threshold)
        )
        if candidates.size < minimum:
            continue
        candidates += history
        order = np.lexsort((candidates, -rises[candidates, frequency]))
        selected = candidates[order[:cap]]
        if selected.size < minimum:
            continue
        mask[selected, frequency] = True
        counts[frequency] = selected.size

    return OnsetResult(np.where(mask[:, :, None], x, 0), mask, rises, counts)
