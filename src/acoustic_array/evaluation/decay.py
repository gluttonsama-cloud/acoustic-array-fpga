"""Schroeder backward energy decay, descriptive T20/T30 (not ISO certification)."""

import numpy as np
from numpy.typing import ArrayLike

from acoustic_array.core.config import positive_int


def decay_summary(impulse: ArrayLike, sample_rate_hz: int) -> dict:
    """Fit -5..-25/-35 dB; return explicit failure for unresolved decay.

    No noise-floor correction: this diagnostic is for noise-free simulated RIRs.
    Fit must leave 10 dB of observed integrated decay before the final nonzero sample.
    """
    positive_int(sample_rate_hz, "sample_rate_hz")
    raw = np.asarray(impulse)
    if np.iscomplexobj(raw):
        raise ValueError("RIR must be real")
    h = np.asarray(raw, dtype=float)
    if h.ndim != 1 or not h.size or not np.isfinite(h).all():
        raise ValueError("Expected finite nonempty RIR vector")
    scale = np.max(np.abs(h))
    if scale == 0:
        return {
            "status": "zero_energy",
            "tail_last_10_percent_energy_db": None,
            "t20": {"status": "zero_energy", "rt60_s": None},
            "t30": {"status": "zero_energy", "rt60_s": None},
        }
    energy = np.cumsum((h[::-1] / scale) ** 2)[::-1]
    positive = energy > 0
    levels = np.full(h.size, -np.inf)
    levels[positive] = 10 * np.log10(energy[positive] / energy[0])
    result = {
        "status": "ok",
        "tail_last_10_percent_energy_db": float(
            10 * np.log10(max(energy[int(0.9 * len(h))] / energy[0], np.finfo(float).tiny))
        ),
    }
    for label, end in [("t20", -25), ("t30", -35)]:
        indices = np.flatnonzero((levels <= -5) & (levels >= end))
        if len(indices) < 10 or levels[positive][-1] > end - 10:
            result[label] = {"status": "insufficient_decay", "rt60_s": None}
            continue
        t = indices / sample_rate_hz
        y = levels[indices]
        slope, intercept = np.polyfit(t, y, 1)
        residual = np.sum((y - (slope * t + intercept)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        if slope >= 0 or total <= 0:
            result[label] = {"status": "nondecaying", "rt60_s": None}
            continue
        result[label] = {
            "status": "ok",
            "rt60_s": float(-60 / slope),
            "r_squared": float(1 - residual / total),
            "fit_start_s": float(t[0]),
            "fit_end_s": float(t[-1]),
            "slope_db_per_s": float(slope),
        }
    return result
