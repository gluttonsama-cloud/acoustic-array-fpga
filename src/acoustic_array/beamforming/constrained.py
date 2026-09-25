"""Fixed geometric minimum-norm constraints (LCMV with identity covariance).

No adaptive covariance estimation. Rank/conditioning and WNG protections
are explicit; fallback is reported rather than hidden as a successful null.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real
from acoustic_array.core.geometry import steering_vectors


@dataclass(frozen=True)
class ConstraintResult:
    weights: NDArray[np.complex128]
    condition_number: NDArray[np.float64]
    constraint_residual: NDArray[np.float64]
    white_noise_gain_db: NDArray[np.float64]
    status: NDArray[np.str_]


def constrained_weights(
    array: ArrayConfig,
    stft: STFTConfig,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
    band_hz: tuple[float, float] = (500, 5000),
    max_condition: float = 1e4,
    min_white_noise_gain_db: float = -10,
) -> ConstraintResult:
    low, high = band_hz
    if not 0 < low < high < array.sample_rate_hz / 2:
        raise ValueError("Constraint band must be strictly within (0, Nyquist)")
    if finite_real(max_condition, "max_condition") < 1:
        raise ValueError("max_condition must be >= 1")
    finite_real(min_white_noise_gain_db, "min_white_noise_gain_db")
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    active_bins = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    if not active_bins.size:
        raise ValueError("Constraint band contains no FFT bins for this fft_size")
    c = steering_vectors(array, frequencies, angles_deg, source_distances_m=source_distances_m)
    outputs = c.shape[2]
    if outputs > array.microphone_count:
        raise ValueError("More constraints than microphones")
    weights = das_weights(array, stft, angles_deg, source_distances_m=source_distances_m)
    condition = np.full(len(frequencies), np.nan)
    residual = np.full((len(frequencies), outputs), np.nan)
    wng = np.full((len(frequencies), outputs), np.nan)
    status = np.full((len(frequencies), outputs), "outside_band_das", dtype="U32")
    response = np.eye(outputs, dtype=np.complex128)
    for f in active_bins:
        # Solve C^H W = I by SVD minimum norm; avoids squaring condition via C^H C.
        candidate, _, rank, singular_values = np.linalg.lstsq(c[f].conj().T, response, rcond=None)
        condition[f] = (
            singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else np.inf
        )
        if rank < outputs or condition[f] > max_condition:
            status[f] = "fallback_rank_or_condition"
        else:
            for k in range(outputs):
                norm_sq = float(np.vdot(candidate[:, k], candidate[:, k]).real)
                candidate_wng = -10 * np.log10(norm_sq)
                if (
                    not np.isfinite(candidate[:, k]).all()
                    or candidate_wng < min_white_noise_gain_db
                ):
                    status[f, k] = "fallback_white_noise_gain"
                else:
                    weights[f, :, k] = candidate[:, k]
                    status[f, k] = "constrained"
        achieved = c[f].conj().T @ weights[f]
        residual[f] = np.max(np.abs(achieved - response), axis=0)
        # General WNG includes actual target gain, valid for both constrained and DAS fallback.
        norm_sq = np.sum(np.abs(weights[f]) ** 2, axis=0)
        wng[f] = 10 * np.log10(np.abs(np.diag(achieved)) ** 2 / norm_sq)
    return ConstraintResult(weights, condition, residual, wng, status)
