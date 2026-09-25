"""Physical source-component ratios, distinct from projection-based BSS metrics."""

import numpy as np
from numpy.typing import ArrayLike


def energy_ratio_db(numerator: float, denominator: float) -> dict:
    """JSON-safe power ratio with explicit silence and zero-denominator states."""
    if not np.isfinite([numerator, denominator]).all() or min(numerator, denominator) < 0:
        raise ValueError("Energies must be finite and nonnegative")
    if numerator == 0:
        return {"db": None, "status": "undefined" if denominator == 0 else "negative_infinity"}
    if denominator == 0:
        return {"db": None, "status": "positive_infinity"}
    return {"db": float(10 * (np.log10(numerator) - np.log10(denominator))), "status": "finite"}


def component_ratios(target: ArrayLike, interference: ArrayLike, noise: ArrayLike) -> dict:
    """Whole provided interval, no demeaning/VAD; interference is the summed waveform."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (target, interference, noise)]
    if any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Components must be finite nonempty vectors")
    if any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Component shapes must match")
    with np.errstate(over="ignore"):
        powers = [float(np.mean(x * x)) for x in arrays]
    if not np.isfinite(powers).all():
        raise ValueError("Component power overflow")
    t, i, n = powers
    return {
        "target_power": t,
        "interference_power": i,
        "noise_power": n,
        "sir": energy_ratio_db(t, i),
        "snr": energy_ratio_db(t, n),
    }


def spectral_power_partition(signal: ArrayLike, sample_rate_hz: int) -> dict:
    """Parseval-exact rectangular-DFT partitions; diagnostic, not the bandpass metric."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty vector")
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 10000
    ):
        raise ValueError("sample_rate_hz must be an integer above 10000")
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum / x.size) ** 2
    power[1:] *= 2
    if x.size % 2 == 0:
        power[-1] /= 2
    f = np.fft.rfftfreq(x.size, 1 / sample_rate_hz)
    if not np.isfinite(power).all():
        raise ValueError("Spectral power overflow")
    return {
        "below_500_hz": float(power[f < 500].sum()),
        "500_to_5000_hz": float(power[(f >= 500) & (f <= 5000)].sum()),
        "above_5000_hz": float(power[f > 5000].sum()),
    }
