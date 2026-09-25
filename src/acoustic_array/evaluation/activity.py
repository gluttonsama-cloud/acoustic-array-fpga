"""Offline reference-only energy activity; not ITU P.56 or a speech recognizer."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import finite_real, positive_int
from acoustic_array.evaluation.metrics import comparison_metrics


@dataclass(frozen=True)
class ActivityConfig:
    frame_ms: float = 20
    hop_ms: float = 10
    hangover_ms: float = 50
    relative_db: float = -40
    absolute_rms: float = 1e-5
    minimum_active_ms: float = 50

    def __post_init__(self) -> None:
        for key in self.__dataclass_fields__:
            finite_real(getattr(self, key), key)
        if not 0 < self.hop_ms <= self.frame_ms or self.hangover_ms < 0:
            raise ValueError("Require 0 < hop <= frame and nonnegative hangover")
        if self.relative_db > 0 or self.absolute_rms <= 0 or self.minimum_active_ms <= 0:
            raise ValueError("Invalid activity thresholds")


_DEFAULT_CONFIG = ActivityConfig()


def reference_activity(
    reference: ArrayLike, sample_rate_hz: int, config: ActivityConfig = _DEFAULT_CONFIG
) -> NDArray[np.bool_]:
    """Frame RMS on globally centered reference; cover active frames plus hangover."""
    positive_int(sample_rate_hz, "sample_rate_hz")
    raw = np.asarray(reference)
    if np.iscomplexobj(raw):
        raise ValueError("Reference must be real")
    x = np.asarray(raw, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty reference vector")
    frame = round(config.frame_ms * sample_rate_hz / 1000)
    hop = round(config.hop_ms * sample_rate_hz / 1000)
    tail = round(config.hangover_ms * sample_rate_hz / 1000)
    if min(frame, hop) < 1:
        raise ValueError("Frame/hop round to zero samples")
    x = x - x.mean()
    starts = np.arange(0, len(x), hop)
    with np.errstate(over="ignore", invalid="ignore"):
        rms = np.array([np.sqrt(np.mean(x[s : s + frame] ** 2)) for s in starts])
    if not np.isfinite(rms).all():
        raise ValueError("Activity energy overflow")
    threshold = max(config.absolute_rms, float(rms.max()) * 10 ** (config.relative_db / 20))
    mask = np.zeros(len(x), dtype=bool)
    for start in starts[rms >= threshold]:
        mask[start : min(len(x), start + frame + tail)] = True
    return mask


def masked_comparison(
    reference: ArrayLike,
    baseline: ArrayLike,
    estimate: ArrayLike,
    mask: ArrayLike,
    sample_rate_hz: int,
    config: ActivityConfig = _DEFAULT_CONFIG,
) -> dict:
    """Apply a precomputed, same-length mask; filter and crop before calling."""
    positive_int(sample_rate_hz, "sample_rate_hz")
    arrays = [np.asarray(x) for x in [reference, baseline, estimate]]
    m = np.asarray(mask)
    if (
        any(x.ndim != 1 or np.iscomplexobj(x) or not np.isfinite(x).all() for x in arrays)
        or any(x.shape != arrays[0].shape for x in arrays)
        or m.shape != arrays[0].shape
        or m.dtype != np.bool_
    ):
        raise ValueError("Expected same-length finite vectors and boolean mask")
    count = int(m.sum())
    minimum = max(2, int(np.ceil(config.minimum_active_ms * sample_rate_hz / 1000)))
    if count < minimum:
        return {"status": "insufficient_activity", "active_samples": count, "metrics": None}
    if np.ptp(arrays[0][m]) == 0:
        return {"status": "silent_reference", "active_samples": count, "metrics": None}
    return {
        "status": "ok",
        "active_samples": count,
        "metrics": comparison_metrics(*[x[m] for x in arrays]),
    }
