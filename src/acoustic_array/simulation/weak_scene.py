"""Offline two-source level calibration on a fixed target-event scoring mask."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import finite_real


@dataclass(frozen=True)
class WeakScene:
    target: NDArray[np.float64]
    interference: NDArray[np.float64]
    target_gain: float
    interference_gain: float
    measured_sir_db: float


def calibrate_weak_scene(
    target: ArrayLike,
    interference: ArrayLike,
    mask: ArrayLike,
    *,
    reference_microphone: int,
    target_rms: float,
    sir_db: float,
) -> WeakScene:
    """Keep full reverberant components; calibrate MIC reference on target mask.

    SIR means full target power / full interfering-source power on exactly the
    same target-event samples. This offline truth must not enter beamforming.
    No per-channel normalization, clipping, noise addition, or mask adaptation.
    """
    raw = [np.asarray(x) for x in (target, interference)]
    m = np.asarray(mask)
    level = finite_real(target_rms, "target_rms")
    ratio = finite_real(sir_db, "sir_db")
    if not 0 < level < 1 or not -60 <= ratio <= 60:
        raise ValueError("Require 0 < target_rms < 1 and -60 <= sir_db <= 60")
    if (
        any(x.ndim != 2 or not x.size or np.iscomplexobj(x) for x in raw)
        or raw[0].shape != raw[1].shape
        or m.dtype != np.bool_
        or m.shape != (raw[0].shape[0],)
        or m.sum() < 2
        or isinstance(reference_microphone, bool)
        or not isinstance(reference_microphone, int)
        or not 0 <= reference_microphone < raw[0].shape[1]
    ):
        raise ValueError("Expected matching real [sample,mic], boolean mask and valid reference")
    arrays = [np.asarray(x, dtype=float) for x in raw]
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Components must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        rms = np.array([np.sqrt(np.mean(x[m, reference_microphone] ** 2)) for x in arrays])
    if not np.isfinite(rms).all() or np.any(rms <= np.finfo(float).tiny):
        raise ValueError("Both components need finite positive energy on target mask")
    gains = np.array([level, level * 10 ** (-ratio / 20)]) / rms
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = [x * gain for x, gain in zip(arrays, gains, strict=True)]
    if any(not np.isfinite(x).all() for x in scaled):
        raise ValueError("Scaled component overflow")
    powers = [np.mean(x[m, reference_microphone] ** 2) for x in scaled]
    actual = float(10 * np.log10(powers[0] / powers[1]))
    if not np.isfinite(actual):
        raise ValueError("Calibrated ratio is nonfinite")
    return WeakScene(scaled[0], scaled[1], float(gains[0]), float(gains[1]), actual)
