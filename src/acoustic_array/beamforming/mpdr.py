"""Mixture-covariance MPDR with trace-relative diagonal loading.

The target is present in training data: this is MPDR, not target-free MVDR.
Calibration is explicit and disjoint from evaluation; weights are then frozen.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.beamforming.constrained import ConstraintResult
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real, positive_int
from acoustic_array.core.geometry import steering_vectors
from acoustic_array.dsp.stft import StreamingSTFT, pcm_array


def mixture_covariance(
    pcm: ArrayLike, array: ArrayConfig, stft: STFTConfig, *, block_size: int = 777
) -> tuple[NDArray[np.complex128], int]:
    """Average xx^H over complete, unpadded frames only; no source masks."""
    positive_int(block_size, "block_size")
    data = pcm_array(pcm, array.microphone_count)
    analyzer = StreamingSTFT(stft, array.microphone_count)
    covariance = np.zeros(
        (stft.fft_size // 2 + 1, array.microphone_count, array.microphone_count), complex
    )
    count = 0
    for start in range(0, len(data), block_size):
        for frame in analyzer.push(data[start : start + block_size], sample_start=start):
            if frame.sample_start < 0:
                continue
            x = frame.spectrum
            covariance += x[:, :, None] * x[:, None, :].conj()
            count += 1
    if count < array.microphone_count:
        raise ValueError("Calibration needs at least microphone_count complete frames")
    covariance /= count
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance overflow")
    return covariance, count


def mpdr_weights(
    covariance: ArrayLike,
    array: ArrayConfig,
    stft: STFTConfig,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
    diagonal_loading: float = 0.1,
    band_hz: tuple[float, float] = (100, 5000),
    max_condition: float = 1e6,
    min_white_noise_gain_db: float = -10,
) -> ConstraintResult:
    """Solve (R + loading*trace(R)/M*I)u=a; w=u/(a^H u).

    Each output has one distortionless direct-path constraint; other target
    directions are not hard nulls. Degenerate/protected bins fall back to DAS.
    Residual is |a_k^H w_k - 1| (not the fixed method's full C^H W-I residual).
    """
    loading = finite_real(diagonal_loading, "diagonal_loading")
    if loading <= 0 or finite_real(max_condition, "max_condition") < 1:
        raise ValueError("Require positive loading and max_condition >= 1")
    finite_real(min_white_noise_gain_db, "min_white_noise_gain_db")
    low, high = band_hz
    if not 0 < low < high < array.sample_rate_hz / 2:
        raise ValueError("Invalid processing band")
    r = np.asarray(covariance, dtype=complex)
    shape = (stft.fft_size // 2 + 1, array.microphone_count, array.microphone_count)
    if r.shape != shape or not np.isfinite(r).all():
        raise ValueError("Expected finite covariance [frequency,mic,mic]")
    scale = np.max(np.abs(r), axis=(1, 2))
    tolerance = 1e-10 * np.maximum(scale, np.finfo(float).tiny)
    if np.any(np.max(np.abs(r - r.conj().transpose(0, 2, 1)), axis=(1, 2)) > tolerance):
        raise ValueError("Covariance must be Hermitian")
    r = (r + r.conj().transpose(0, 2, 1)) / 2
    if np.any(np.linalg.eigvalsh(r)[:, 0] < -tolerance):
        raise ValueError("Covariance must be positive semidefinite")
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    bins = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    if not bins.size:
        raise ValueError("Band contains no FFT bins")
    a = steering_vectors(array, frequencies, angles_deg, source_distances_m=source_distances_m)
    weights = das_weights(array, stft, angles_deg, source_distances_m=source_distances_m)
    outputs = weights.shape[2]
    condition = np.full(len(frequencies), np.nan)
    residual = np.full((len(frequencies), outputs), np.nan)
    wng = residual.copy()
    status = np.full(residual.shape, "outside_band_das", dtype="U32")
    for f in bins:
        power = float(np.trace(r[f]).real / array.microphone_count)
        if power <= np.finfo(float).tiny:
            status[f] = "fallback_zero_power"
        else:
            loaded = r[f] / power + loading * np.eye(array.microphone_count)
            condition[f] = np.linalg.cond(loaded)
            if condition[f] > max_condition:
                status[f] = "fallback_condition"
            else:
                u = np.linalg.solve(loaded, a[f])
                gain = np.sum(a[f].conj() * u, axis=0)
                candidate = u / gain
                for k in range(outputs):
                    norm = float(np.vdot(candidate[:, k], candidate[:, k]).real)
                    if (
                        not np.isfinite(candidate[:, k]).all()
                        or norm <= 0
                        or -10 * np.log10(norm) < min_white_noise_gain_db
                    ):
                        status[f, k] = "fallback_white_noise_gain"
                    else:
                        weights[f, :, k] = candidate[:, k]
                        status[f, k] = "mpdr"
        response = np.sum(a[f].conj() * weights[f], axis=0)
        residual[f] = np.abs(response - 1)
        wng[f] = 10 * np.log10(np.abs(response) ** 2 / np.sum(np.abs(weights[f]) ** 2, axis=0))
    return ConstraintResult(weights, condition, residual, wng, status)
