"""ULA geometry: positive angles point toward increasing microphone x."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.core.config import ArrayConfig


def microphone_positions(config: ArrayConfig) -> NDArray[np.float64]:
    if config.positions_m is not None:
        return np.array(config.positions_m, dtype=float)
    positions = np.zeros((config.microphone_count, 3), dtype=np.float64)
    positions[:, 0] = (
        np.arange(config.microphone_count) - (config.microphone_count - 1) / 2
    ) * config.spacing_m
    return positions


def directions(angles_deg: ArrayLike) -> NDArray[np.float64]:
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or not angles.size or not np.isfinite(angles).all():
        raise ValueError("angles_deg must be a nonempty finite vector")
    if np.any(np.abs(angles) > 90):
        raise ValueError("M1 directions are restricted to the front half-plane [-90, 90]")
    radians = np.deg2rad(angles)
    return np.column_stack((np.sin(radians), np.cos(radians), np.zeros_like(radians)))


def relative_delays(config: ArrayConfig, angles_deg: ArrayLike) -> NDArray[np.float64]:
    """Return [microphone, direction] delays relative to the physical reference mic."""
    position = microphone_positions(config)
    relative = position - position[config.reference_microphone_index]
    return -(relative @ directions(angles_deg).T) / config.sound_speed_m_per_s


def steering_vectors(
    config: ArrayConfig,
    frequencies_hz: ArrayLike,
    angles_deg: ArrayLike,
    *,
    source_distances_m: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all():
        raise ValueError("frequencies_hz must be a nonempty finite vector")
    if np.any(frequencies < 0) or np.any(frequencies > config.sample_rate_hz / 2):
        raise ValueError("Frequency outside [0, Nyquist]")
    if source_distances_m is None:
        delays = relative_delays(config, angles_deg)
        amplitude = 1.0
    else:
        vectors = directions(angles_deg)
        raw_distances = np.asarray(source_distances_m, dtype=object)
        if any(
            isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in raw_distances.flat
        ):
            raise ValueError("Distances must be real numbers, not booleans or complex values")
        distances = np.asarray(source_distances_m, dtype=float)
        if (
            distances.shape != (len(vectors),)
            or not np.isfinite(distances).all()
            or np.any(distances <= 0)
        ):
            raise ValueError("One finite positive center distance is required per source")
        sources = vectors * distances[:, None]
        paths = np.linalg.norm(microphone_positions(config)[:, None, :] - sources, axis=2)
        if np.any(paths < 1e-6):
            raise ValueError("Source must be at least one micrometre from each microphone")
        reference = paths[config.reference_microphone_index]
        delays = (paths - reference) / config.sound_speed_m_per_s
        amplitude = reference / paths
    return amplitude * np.exp(-2j * np.pi * frequencies[:, None, None] * delays[None, :, :])
