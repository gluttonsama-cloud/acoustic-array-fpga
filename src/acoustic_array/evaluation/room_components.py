"""Physical replay diagnostics; coherent direct/reflected power is not additive."""

import numpy as np
from numpy.typing import ArrayLike

from acoustic_array.evaluation.components import component_ratios, energy_ratio_db
from acoustic_array.evaluation.metrics import comparison_metrics


def target_component_summary(
    reference: ArrayLike,
    direct: ArrayLike,
    reverberant_target: ArrayLike,
    interference: ArrayLike,
    noise: ArrayLike,
    baseline: ArrayLike,
) -> dict:
    """Inputs must already share the same evaluation filter/crop/activity mask.

    Target total includes its own reflections; interference is all other sources
    summed as waveforms, not a sum of independent powers. No alignment search.
    """
    raw = [
        np.asarray(x)
        for x in [reference, direct, reverberant_target, interference, noise, baseline]
    ]
    if any(np.iscomplexobj(x) or x.ndim != 1 or x.size < 2 for x in raw):
        raise ValueError("Expected real nontrivial component vectors")
    arrays = [np.asarray(x, dtype=float) for x in raw]
    if any(x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Expected same-length finite components")
    ref, d, target, other, n, mix = arrays
    reflection = target - d

    def power(x):
        value = float(np.mean(x * x))
        if not np.isfinite(value):
            raise ValueError("Component power overflow")
        return value

    centered_ref = ref - ref.mean()
    denominator = float(centered_ref @ centered_ref)
    if denominator <= 0:
        raise ValueError("Reference must have nonzero centered energy")
    centered_direct = d - d.mean()
    gain = float(centered_direct @ centered_ref / denominator)
    residual = centered_direct - gain * centered_ref
    return {
        "direct_projection_gain": gain,
        "direct_error_to_reference": energy_ratio_db(power(residual), power(centered_ref)),
        "direct_power": power(d),
        "reflection_power": power(reflection),
        "direct_reflection_cross_power": float(2 * np.mean(d * reflection)),
        "target_total_power": power(target),
        "reflection_to_direct": energy_ratio_db(power(reflection), power(d)),
        "total_target_ratios": component_ratios(target, other, n),
        "direct_to_other_sources": energy_ratio_db(power(d), power(other)),
        "target_only_vs_direct_reference": comparison_metrics(ref, mix, target),
        "full_output_vs_direct_reference": comparison_metrics(ref, mix, target + other + n),
    }
