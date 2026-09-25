"""SI-SDR with declared zero-mean projection; no alignment search or gain tuning."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, sosfiltfilt


def comparison_metrics(reference: ArrayLike, baseline: ArrayLike, estimate: ArrayLike) -> dict:
    """Serialize non-finite scores explicitly; never fabricate a finite ceiling."""
    input_score = si_sdr(reference, baseline)
    output_score = si_sdr(reference, estimate)
    improvement = output_score - input_score
    scores = {
        "input_si_sdr_db": input_score,
        "output_si_sdr_db": output_score,
        "si_sdri_db": improvement,
    }
    status = {}
    result = {}
    for name, value in scores.items():
        result[name] = value if np.isfinite(value) else None
        if np.isnan(value):
            status[name] = "undefined_infinite_difference"
        elif value == np.inf:
            status[name] = "positive_infinity"
        elif value == -np.inf:
            status[name] = "negative_infinity"
        else:
            status[name] = "finite"
    result["metric_status"] = status
    return result


def si_sdr(reference: ArrayLike, estimate: ArrayLike) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.ndim != 1 or reference.shape != estimate.shape or reference.size < 2:
        raise ValueError("Expected equally sized nontrivial one-dimensional signals")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("SI-SDR inputs must be finite")
    target = reference - reference.mean()
    output = estimate - estimate.mean()
    target_energy = float(target @ target)
    if target_energy == 0:
        raise ValueError("SI-SDR undefined for a silent/constant reference")
    projection = target * (float(output @ target) / target_energy)
    error = output - projection
    projected_energy = float(projection @ projection)
    error_energy = float(error @ error)
    if projected_energy == 0:
        return float("-inf")
    if error_energy == 0:
        return float("inf")
    return float(10 * np.log10(projected_energy / error_energy))


def bandpass_for_evaluation(
    pcm: ArrayLike, sample_rate_hz: int, band_hz: tuple[float, float]
) -> NDArray[np.float64]:
    data = np.asarray(pcm, dtype=np.float64)
    if data.ndim not in (1, 2) or not np.isfinite(data).all():
        raise ValueError("Expected finite PCM with sample axis first")
    low, high = band_hz
    if not 0 < low < high < sample_rate_hz / 2:
        raise ValueError("Invalid evaluation frequency band")
    sos = butter(6, [low, high], fs=sample_rate_hz, btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=0)
