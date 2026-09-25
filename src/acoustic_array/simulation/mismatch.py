"""Offline static channel gain and residual-delay mismatch, before processing."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.simulation.free_field import fractional_delay


def apply_channel_mismatch(
    pcm: ArrayLike, gain_db: ArrayLike, delay_samples: ArrayLike
) -> NDArray[np.float64]:
    """Apply one gain/delay per channel to [...,sample,channel], with zero extension.

    Positive delay means later arrival. The same transfer must apply to source
    components and pre-transfer noise; post-transfer electronics noise is not modeled.
    """
    raw = np.asarray(pcm)
    if np.iscomplexobj(raw):
        raise ValueError("PCM must be real")
    data = np.asarray(raw, dtype=np.float64)
    gains, delays = np.asarray(gain_db), np.asarray(delay_samples)
    if np.iscomplexobj(gains) or np.iscomplexobj(delays):
        raise ValueError("Channel parameters must be real")
    gains, delays = gains.astype(float), delays.astype(float)
    if data.ndim not in (2, 3) or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite nonempty [sample,channel] or [source,sample,channel]")
    if gains.shape != (data.shape[-1],) or delays.shape != gains.shape:
        raise ValueError("One gain and delay required per channel")
    if not np.isfinite(gains).all() or not np.isfinite(delays).all():
        raise ValueError("Channel parameters must be finite")
    # Explicit development limits prevent pathological transfer values/large shifts.
    if np.any(np.abs(gains) > 20) or np.any(np.abs(delays) > 4):
        raise ValueError("Development mismatch limits: gain +/-20 dB, delay +/-4 samples")
    result = np.empty_like(data)
    frames = data[None] if data.ndim == 2 else data
    outputs = result[None] if data.ndim == 2 else result
    for source, output in zip(frames, outputs, strict=True):
        for mic in range(data.shape[-1]):
            shifted = (
                source[:, mic]
                if delays[mic] == 0
                else fractional_delay(source[:, mic], float(delays[mic]))
            )
            output[:, mic] = shifted * 10 ** (gains[mic] / 20)
    if not np.isfinite(result).all():
        raise ValueError("Mismatch output overflow")
    return result
