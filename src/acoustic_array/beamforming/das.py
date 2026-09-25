"""Conventional equal-gain delay-and-sum baseline, with known steering directions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import steering_vectors


def das_weights(
    array: ArrayConfig,
    stft: STFTConfig,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    manifold = steering_vectors(
        array, frequencies, angles_deg, source_distances_m=source_distances_m
    )
    weights = manifold / (
        array.microphone_count
        if source_distances_m is None
        else np.sum(np.abs(manifold) ** 2, axis=1, keepdims=True)
    )
    # A real signal has real DC/Nyquist coefficients. Fractional delay at Nyquist
    # cannot retain an arbitrary complex phase in a real, even-length DFT.
    weights[[0, -1]] = weights[[0, -1]].real
    return weights


def apply_weights(spectrum: ArrayLike, weights: ArrayLike) -> NDArray[np.complex128]:
    x = np.asarray(spectrum, dtype=np.complex128)
    w = np.asarray(weights, dtype=np.complex128)
    if x.ndim != 2 or w.ndim != 3 or x.shape != w.shape[:2] or not w.shape[2]:
        raise ValueError("Expected spectrum [frequency,mic] and weights [frequency,mic,output]")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("Spectrum and weights must be finite")
    result = np.einsum("fm,fmk->fk", x, w.conj(), optimize=True)
    result[[0, -1]] = result[[0, -1]].real
    return result
