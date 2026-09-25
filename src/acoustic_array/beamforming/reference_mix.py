"""Aligned reference mixing; a trade-off, not a target-preservation guarantee."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mix_reference(output: ArrayLike, reference: ArrayLike, fraction: float) -> NDArray[np.float64]:
    """Mix [sample,port] output and same-index [sample] reference; no gain fitting."""
    y, x = np.asarray(output), np.asarray(reference)
    if (
        y.ndim != 2
        or x.ndim != 1
        or y.shape[0] != x.size
        or not y.shape[1]
        or np.iscomplexobj(y)
        or np.iscomplexobj(x)
        or not np.isfinite(y).all()
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected aligned finite real output and reference")
    if (
        isinstance(fraction, (bool, np.bool_))
        or not isinstance(fraction, (int, float))
        or not np.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("fraction must be a finite real value in [0,1]")
    return (1 - fraction) * y.astype(float) + fraction * x.astype(float)[:, None]
