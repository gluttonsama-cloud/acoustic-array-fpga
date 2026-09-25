"""Reference-mask active-RMS normalization and calibrated S1-like noise."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig, finite_real
from acoustic_array.evaluation.activity import ActivityConfig, reference_activity
from acoustic_array.simulation.free_field import SimulatedScene, simulate_far_field
from acoustic_array.simulation.near_field import simulate_near_field


@dataclass(frozen=True)
class ActiveScene:
    scene: SimulatedScene
    masks: NDArray[np.bool_]
    source_gains: NDArray[np.float64]
    active_rms: NDArray[np.float64]
    reference_noise_rms: float
    measured_active_snr_db: NDArray[np.float64]


def build_active_scene(
    sources: ArrayLike,
    array: ArrayConfig,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
    active_rms: float = 0.04,
    noise_snr_db: float = 20,
    seed: int = 0,
    trim_samples: int = 2400,
) -> ActiveScene:
    """Freeze pre-gain masks; normalize propagated MIC reference on cropped activity.

    Noise is globally scaled to exact reference-channel RMS over the full cropped
    interval. Active-window SNR is measured separately, not forced per speaker.
    No signal clipping or individual output normalization is performed.
    """
    if isinstance(trim_samples, bool) or not isinstance(trim_samples, int) or trim_samples < 0:
        raise ValueError("trim_samples must be a nonnegative integer")
    raw = np.asarray(sources)
    if np.iscomplexobj(raw):
        raise ValueError("Sources must be real")
    data = np.asarray(raw, dtype=float)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite nonempty [source,sample]")
    if 2 * trim_samples >= data.shape[1]:
        raise ValueError("Trim removes all samples")
    data = data - data.mean(axis=1, keepdims=True)
    base = (
        simulate_far_field(data, array, angles_deg)
        if source_distances_m is None
        else simulate_near_field(data, array, angles_deg, source_distances_m)
    )
    return normalize_active_scene(
        base,
        array,
        active_rms=active_rms,
        noise_snr_db=noise_snr_db,
        seed=seed,
        trim_samples=trim_samples,
    )


def normalize_active_scene(
    base: SimulatedScene,
    array: ArrayConfig,
    *,
    activity_components: ArrayLike | None = None,
    active_rms: float = 0.04,
    noise_snr_db: float = 20,
    seed: int = 0,
    trim_samples: int = 2400,
) -> ActiveScene:
    """Scale a noiseless scene using fixed reference components.

    Room experiments supply direct-only components for masks/levels; scaling is
    applied to the entire reverberant source, preserving direct/reverberant ratio.
    Existing noise is deliberately forbidden, not silently discarded.
    """
    rms_target = finite_real(active_rms, "active_rms")
    snr = finite_real(noise_snr_db, "noise_snr_db")
    if not 0 < rms_target <= 1 or not -40 <= snr <= 100:
        raise ValueError("Require 0 < active_rms <= 1 and -40 <= noise_snr_db <= 100")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid seed")
    if isinstance(trim_samples, bool) or not isinstance(trim_samples, int) or trim_samples < 0:
        raise ValueError("trim_samples must be a nonnegative integer")
    components = np.asarray(base.components)
    references = np.asarray(activity_components) if activity_components is not None else components
    if (
        components.ndim != 3
        or not components.size
        or components.shape[2] != array.microphone_count
        or references.shape != components.shape
        or np.iscomplexobj(components)
        or np.iscomplexobj(references)
        or not np.isfinite(components).all()
        or not np.isfinite(references).all()
        or base.noise.shape != components.shape[1:]
        or np.any(base.noise != 0)
    ):
        raise ValueError("Expected noiseless finite real [source,sample,mic] components")
    if 2 * trim_samples >= components.shape[1]:
        raise ValueError("Trim removes all samples")
    refs = references[:, :, array.reference_microphone_index]
    masks = np.stack([reference_activity(x, array.sample_rate_hz) for x in refs])
    masks[:, :trim_samples] = False
    if trim_samples:
        masks[:, -trim_samples:] = False
    minimum = max(2, int(np.ceil(ActivityConfig().minimum_active_ms * array.sample_rate_hz / 1000)))
    gains = []
    for reference, mask in zip(refs, masks, strict=True):
        if mask.sum() < minimum:
            raise ValueError("Insufficient reference activity")
        measured = float(np.sqrt(np.mean(reference[mask] ** 2)))
        if not np.isfinite(measured) or measured <= 0:
            raise ValueError("Invalid active energy")
        gains.append(rms_target / measured)
    gains = np.asarray(gains)
    components = base.components * gains[:, None, None]
    desired_noise = rms_target * 10 ** (-snr / 20)
    noise = np.random.default_rng(seed).normal(size=base.noise.shape)
    crop = slice(trim_samples, -trim_samples if trim_samples else None)
    ref = array.reference_microphone_index
    noise *= desired_noise / np.sqrt(np.mean(noise[crop, ref] ** 2))
    mixture = components.sum(axis=0) + noise
    if not np.isfinite(mixture).all():
        raise ValueError("Scene overflow")
    actual = np.array(
        [np.sqrt(np.mean((x[m] * g) ** 2)) for x, m, g in zip(refs, masks, gains, strict=True)]
    )
    active_noise = np.array([np.sqrt(np.mean(noise[m, ref] ** 2)) for m in masks])
    scene = SimulatedScene(mixture, components, noise, base.relative_delays_samples)
    return ActiveScene(
        scene,
        masks,
        gains,
        actual,
        float(np.sqrt(np.mean(noise[crop, ref] ** 2))),
        20 * np.log10(actual / active_noise),
    )
