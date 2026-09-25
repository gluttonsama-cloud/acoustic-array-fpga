"""Loaded LCMV with three sampled target-angle constraints per output.

The constraints protect only the sampled angles, not a continuous sector.
Each target is solved independently; no other target direction is a null.
"""

import numpy as np
from numpy.typing import ArrayLike

from acoustic_array.beamforming.constrained import ConstraintResult
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real
from acoustic_array.core.geometry import directions, steering_vectors


def sector_lcmv_weights(
    covariance: ArrayLike,
    array: ArrayConfig,
    stft: STFTConfig,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
    half_width_deg: float = 2.0,
    diagonal_loading: float = 0.1,
    band_hz: tuple[float, float] = (100, 5000),
    max_condition: float = 1e6,
    min_white_noise_gain_db: float = -10,
) -> ConstraintResult:
    """Minimize loaded mixture power subject to C^H w = [1, 1, 1].

    C holds steering vectors at center-width, center, center+width. Loading
    uses R/mean(diag(R)) + diagonal_loading*I, as in MPDR. The reported
    condition is max(cond(loaded R), cond(C^H loaded_R^-1 C)) across targets
    for each bin; residual is the largest error across the three constraints.
    Failed or out-of-band bins use center-direction DAS. A DAS fallback does
    not imply that all three constraints or the WNG threshold were met.
    """
    width = finite_real(half_width_deg, "half_width_deg")
    loading = finite_real(diagonal_loading, "diagonal_loading")
    condition_limit = finite_real(max_condition, "max_condition")
    finite_real(min_white_noise_gain_db, "min_white_noise_gain_db")
    if width <= 0 or loading <= 0 or condition_limit < 1:
        raise ValueError("Require positive half_width_deg and loading, max_condition >= 1")
    try:
        low, high = [finite_real(edge, "band edge") for edge in band_hz]
    except (TypeError, ValueError) as exc:
        raise ValueError("band_hz must have two finite real edges") from exc
    if not 0 < low < high < array.sample_rate_hz / 2:
        raise ValueError("Invalid processing band")

    raw_angles = np.asarray(angles_deg, dtype=object)
    if any(
        isinstance(value, (bool, np.bool_, complex, np.complexfloating))
        for value in raw_angles.flat
    ):
        raise ValueError("angles_deg must contain real numbers, not booleans or complex values")
    centers = np.asarray(angles_deg, dtype=float)
    directions(centers)
    sample_angles = (centers[:, None] + np.array([-width, 0.0, width])).ravel()
    # This also rejects a window crossing the front-half-plane boundary.
    directions(sample_angles)
    sample_distances = None
    if source_distances_m is not None:
        sample_distances = np.repeat(np.asarray(source_distances_m, dtype=object), 3)

    r = np.asarray(covariance, dtype=complex)
    frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
    shape = (len(frequencies), array.microphone_count, array.microphone_count)
    if r.shape != shape or not np.isfinite(r).all():
        raise ValueError("Expected finite covariance [frequency,mic,mic]")
    # Normalize before subtraction, symmetrization, eigensolve and trace so
    # finite extreme magnitudes cannot overflow an otherwise scale-free solve.
    scale = np.maximum(np.max(np.abs(r.real), axis=(1, 2)), np.max(np.abs(r.imag), axis=(1, 2)))
    safe_scale = np.where(scale > 0, scale, 1.0)
    r = r.real / safe_scale[:, None, None] + 1j * (r.imag / safe_scale[:, None, None])
    tolerance = np.full(len(frequencies), 1e-10)
    if np.any(np.max(np.abs(r - r.conj().transpose(0, 2, 1)), axis=(1, 2)) > tolerance):
        raise ValueError("Covariance must be Hermitian")
    r = (r + r.conj().transpose(0, 2, 1)) / 2
    try:
        eigenvalues = np.linalg.eigvalsh(r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Covariance eigensolve failed") from exc
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("Covariance must be positive semidefinite")

    bins = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    if not bins.size:
        raise ValueError("Band contains no FFT bins")
    manifold = steering_vectors(
        array, frequencies, sample_angles, source_distances_m=sample_distances
    ).reshape(len(frequencies), array.microphone_count, len(centers), 3)
    weights = das_weights(array, stft, centers, source_distances_m=source_distances_m)
    condition = np.full(len(frequencies), np.nan)
    residual = np.full((len(frequencies), len(centers)), np.nan)
    wng = np.full_like(residual, np.nan)
    status = np.full(residual.shape, "outside_band_das", dtype="U32")
    response = np.ones(3, dtype=complex)
    eye = np.eye(array.microphone_count)

    for f in bins:
        power = float(np.trace(r[f]).real / array.microphone_count)
        if power <= np.finfo(float).tiny:
            status[f] = "fallback_zero_power"
        else:
            loaded = r[f] / power + loading * eye
            try:
                loaded_condition = float(np.linalg.cond(loaded))
            except np.linalg.LinAlgError:
                loaded_condition = np.inf
            condition[f] = loaded_condition
            if not np.isfinite(loaded_condition) or loaded_condition > condition_limit:
                status[f] = "fallback_rank_or_condition"
            else:
                for k in range(len(centers)):
                    c = manifold[f, :, k, :]
                    try:
                        u = np.linalg.solve(loaded, c)
                        gram = c.conj().T @ u
                        singular = np.linalg.svd(gram, compute_uv=False)
                        gram_condition = (
                            float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
                        )
                        condition[f] = max(condition[f], gram_condition)
                        if (
                            singular[-1] <= np.finfo(float).eps * 3 * singular[0]
                            or not np.isfinite(gram_condition)
                            or gram_condition > condition_limit
                        ):
                            status[f, k] = "fallback_rank_or_condition"
                            continue
                        candidate = u @ np.linalg.solve(gram, response)
                        norm_sq = float(np.vdot(candidate, candidate).real)
                        if not np.isfinite(candidate).all() or norm_sq <= 0:
                            status[f, k] = "fallback_rank_or_condition"
                        elif -10 * np.log10(norm_sq) < min_white_noise_gain_db:
                            status[f, k] = "fallback_white_noise_gain"
                        elif np.max(np.abs(c.conj().T @ candidate - response)) > 1e-6:
                            status[f, k] = "fallback_rank_or_condition"
                        else:
                            weights[f, :, k] = candidate
                            status[f, k] = "sector_lcmv"
                    except np.linalg.LinAlgError:
                        status[f, k] = "fallback_rank_or_condition"

        for k in range(len(centers)):
            achieved = manifold[f, :, k, :].conj().T @ weights[f, :, k]
            residual[f, k] = np.max(np.abs(achieved - response))
            norm_sq = float(np.vdot(weights[f, :, k], weights[f, :, k]).real)
            wng[f, k] = 10 * np.log10(np.abs(achieved[1]) ** 2 / norm_sq)
    return ConstraintResult(weights, condition, residual, wng, status)
