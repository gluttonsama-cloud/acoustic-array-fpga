"""Independent direct spherical propagation, referenced to the physical MIC.

Sources specify pressure at the reference microphone, not emitter power.
Distances are from the array centre. No reverberation or absolute flight time.
"""

import math

import numpy as np
from numpy.typing import ArrayLike

from acoustic_array.core.config import ArrayConfig
from acoustic_array.simulation.free_field import SimulatedScene, fractional_delay


def simulate_near_field(
    sources: ArrayLike,
    array: ArrayConfig,
    angles_deg: ArrayLike,
    source_distances_m: ArrayLike,
) -> SimulatedScene:
    raw = np.asarray(sources)
    if np.iscomplexobj(raw):
        raise ValueError("Sources must be real")
    data = np.asarray(raw, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    raw_distances = np.asarray(source_distances_m, dtype=object)
    if any(
        isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in raw_distances.flat
    ):
        raise ValueError("Distances must be real numbers, not booleans or complex values")
    distances = np.asarray(source_distances_m, dtype=float)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite nonempty [source,sample]")
    if angles.shape != (len(data),) or not np.isfinite(angles).all() or np.any(np.abs(angles) > 90):
        raise ValueError("One finite front-half-plane angle required per source")
    if (
        distances.shape != angles.shape
        or not np.isfinite(distances).all()
        or np.any(distances <= 0)
    ):
        raise ValueError("One finite positive center distance required per source")
    components = np.empty((len(data), data.shape[1], array.microphone_count))
    delays = np.empty((len(data), array.microphone_count))
    # Scalar geometry deliberately independent of the steering-vector implementation.
    for k, (angle, distance) in enumerate(zip(angles, distances, strict=True)):
        sx = float(distance) * math.sin(math.radians(float(angle)))
        sy = float(distance) * math.cos(math.radians(float(angle)))
        paths = [
            math.hypot(sx - (m - (array.microphone_count - 1) / 2) * array.spacing_m, sy)
            for m in range(array.microphone_count)
        ]
        if array.positions_m is not None:
            paths = [math.dist((sx, sy, 0.0), position) for position in array.positions_m]
        if min(paths) < 1e-6:
            raise ValueError("Source must be at least one micrometre from each microphone")
        reference = paths[array.reference_microphone_index]
        for m, path in enumerate(paths):
            delay = (path - reference) / array.sound_speed_m_per_s * array.sample_rate_hz
            delays[k, m] = delay
            components[k, :, m] = fractional_delay(data[k], delay) * reference / path
    noise = np.zeros((data.shape[1], array.microphone_count))
    return SimulatedScene(components.sum(axis=0), components, noise, delays)
