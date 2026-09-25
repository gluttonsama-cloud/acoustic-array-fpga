"""Offline shoebox ISM adapter; absolute delays, frequency-independent walls.

Optional rir-generator dependency stays outside the streaming/beamforming layer.
No RIR or source component is exposed to beamforming weight generation.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import fftconvolve

from acoustic_array.core.config import ArrayConfig, finite_real
from acoustic_array.core.geometry import directions, microphone_positions
from acoustic_array.simulation.free_field import SimulatedScene


def _real_vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (size,) or any(
        isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in raw
    ):
        raise ValueError(f"{name} must be a real vector of length {size}")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class RoomConfig:
    dimensions_m: tuple[float, float, float] = (6, 5, 3)
    array_center_m: tuple[float, float, float] = (3, 1.5, 1.5)
    rir_duration_s: float = 0.65

    def __post_init__(self) -> None:
        dimensions = _real_vector(self.dimensions_m, 3, "dimensions_m")
        center = _real_vector(self.array_center_m, 3, "array_center_m")
        if np.any(dimensions <= 0) or np.any(center <= 0) or np.any(center >= dimensions):
            raise ValueError("Room dimensions positive; array center strictly inside room")
        if not 0.05 <= finite_real(self.rir_duration_s, "rir_duration_s") <= 2:
            raise ValueError("RIR duration must be in [0.05, 2] seconds")


@dataclass(frozen=True)
class RoomRIR:
    impulse_responses: NDArray[np.float64]  # [source, tap, mic]
    direct_responses: NDArray[np.float64]
    source_positions_m: NDArray[np.float64]
    microphone_positions_m: NDArray[np.float64]
    direct_distances_m: NDArray[np.float64]  # [source,mic]
    target_rt60_s: float
    energy_absorption: float


def generate_room_rirs(
    array: ArrayConfig,
    angles_deg: ArrayLike,
    source_distances_m: ArrayLike,
    room: RoomConfig,
    target_rt60_s: float,
) -> RoomRIR:
    """Use Habets' rir-generator 0.3.0 with high-pass disabled and all ISM orders.

    Scale each source's RIR by 4*pi*r_ref, so direct MIC reference pressure
    is unity. Unlike relative-delay G1, retain absolute propagation delays.
    Zero target RT60 uses an order-zero control with the identical RIR kernel.
    """
    rt60 = finite_real(target_rt60_s, "target_rt60_s")
    if not 0 <= rt60 <= 1 or (rt60 and room.rir_duration_s < 1.5 * rt60):
        raise ValueError("Require RT60 in [0,1] and RIR duration >= 1.5 * positive RT60")
    vectors = directions(angles_deg)
    distances = _real_vector(source_distances_m, len(vectors), "source_distances_m")
    if np.any(distances <= 0):
        raise ValueError("Distances must be positive")
    center, dimensions = np.asarray(room.array_center_m), np.asarray(room.dimensions_m)
    sources = center + distances[:, None] * vectors
    microphones = center + microphone_positions(array)
    if any(np.any(p <= 0) or np.any(p >= dimensions) for p in [sources, microphones]):
        raise ValueError("Every source and microphone must be strictly inside room")
    paths = np.linalg.norm(sources[:, None, :] - microphones[None, :, :], axis=2)
    if (
        np.min(paths) < 1e-6
        or paths.max() / array.sound_speed_m_per_s + 0.004 >= room.rir_duration_s
    ):
        raise ValueError("Invalid source proximity or RIR too short for direct arrival")
    surface = 2 * (
        dimensions[0] * dimensions[1]
        + dimensions[0] * dimensions[2]
        + dimensions[1] * dimensions[2]
    )
    alpha = (
        (24 * np.log(10) * np.prod(dimensions) / (array.sound_speed_m_per_s * surface * rt60))
        if rt60
        else 1.0
    )
    if not 0 < alpha <= 1:
        raise ValueError("Sabine absorption outside (0,1]; adjust room or target RT60")
    try:
        import rir_generator
    except ImportError as exc:
        raise RuntimeError("Install the room extra: pip install -e .[room]") from exc
    samples = (
        round(min(room.rir_duration_s, 1.5 * rt60 + 0.05) * array.sample_rate_hz)
        if rt60
        else round(room.rir_duration_s * array.sample_rate_hz)
    )

    def generate_one(k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        source = sources[k]
        args = dict(
            c=array.sound_speed_m_per_s,
            fs=array.sample_rate_hz,
            r=microphones,
            s=source,
            L=dimensions,
            hp_filter=False,
            dim=3,
        )
        direct_samples = min(
            samples,
            int(
                np.ceil((paths[k].max() / array.sound_speed_m_per_s + 0.004) * array.sample_rate_hz)
            ),
        )
        short = rir_generator.generate(
            **args, nsample=direct_samples, reverberation_time=0, order=0
        )
        h0 = np.pad(short, ((0, samples - direct_samples), (0, 0)))
        h = (
            rir_generator.generate(**args, nsample=samples, reverberation_time=rt60, order=-1)
            if rt60
            else h0.copy()
        )
        scale = 4 * np.pi * paths[k, array.reference_microphone_index]
        return h * scale, h0 * scale

    # CFFI releases the GIL; cap concurrency to keep CPU/memory use bounded.
    with ThreadPoolExecutor(max_workers=2) as executor:
        pairs = list(executor.map(generate_one, range(len(sources))))
    full, direct = zip(*pairs, strict=True)
    return RoomRIR(
        np.stack(full), np.stack(direct), sources, microphones, paths, rt60, float(alpha)
    )


def propagate_room(
    sources: ArrayLike,
    array: ArrayConfig,
    rirs: RoomRIR,
) -> tuple[SimulatedScene, NDArray[np.float64]]:
    """Convolve with zero initial state; retain first input-length samples.

    Return reverberant components and direct-only components separately. Caller
    must document startup/end crop and use direct references for dereverberation.
    """
    raw = np.asarray(sources)
    if np.iscomplexobj(raw):
        raise ValueError("Sources must be real")
    data = np.asarray(raw, dtype=float)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected finite nonempty [source,sample]")
    h, h0 = rirs.impulse_responses, rirs.direct_responses
    if (
        h.ndim != 3
        or h.shape != h0.shape
        or h.shape[0] != len(data)
        or not h.shape[1]
        or h.shape[2] != array.microphone_count
        or not np.isfinite(h).all()
        or not np.isfinite(h0).all()
    ):
        raise ValueError("Invalid RIR shape or values")
    outputs = []
    for impulse in [h, h0]:
        outputs.append(
            np.stack(
                [
                    fftconvolve(x[:, None], response, mode="full", axes=0)[: data.shape[1]]
                    for x, response in zip(data, impulse, strict=True)
                ]
            )
        )
    noise = np.zeros((data.shape[1], array.microphone_count))
    delays = (
        rirs.direct_distances_m - rirs.direct_distances_m[:, [array.reference_microphone_index]]
    )
    delays = delays / array.sound_speed_m_per_s * array.sample_rate_hz
    return SimulatedScene(outputs[0].sum(axis=0), outputs[0], noise, delays), outputs[1]
