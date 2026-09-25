"""Windowed-sinc propagation simulator for independent far-field checks.

This is an offline scene generator with zero extension, not a real-time
fractional-delay implementation. Negative relative delays mean an earlier
arrival than the reference microphone, not access to future input online.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, fftconvolve, sosfiltfilt

from acoustic_array.core.config import ArrayConfig, finite_real, positive_int


def fractional_delay(
    signal: ArrayLike, delay_samples: float, *, half_taps: int = 64
) -> NDArray[np.float64]:
    raw = np.asarray(signal)
    if np.iscomplexobj(raw):
        raise ValueError("Signal must be real")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite one-dimensional signal")
    delay = finite_real(delay_samples, "delay_samples")
    positive_int(half_taps, "half_taps")
    if not x.size:
        return x.copy()
    integer = int(np.floor(delay))
    fraction = delay - integer
    offsets = np.arange(-half_taps, half_taps + 1)
    kernel = np.sinc(offsets - fraction) * np.kaiser(2 * half_taps + 1, 8.6)
    kernel /= kernel.sum()
    convolution = fftconvolve(x, kernel, mode="full")
    indices = np.arange(x.size) - integer + half_taps
    output = np.zeros_like(x)
    valid = (indices >= 0) & (indices < len(convolution))
    output[valid] = convolution[indices[valid]]
    return output


def synthetic_sources(
    sample_rate_hz: int,
    samples: int,
    count: int,
    seed: int,
    band_hz: tuple[float, float],
) -> NDArray[np.float64]:
    """Independent amplitude-modulated band-limited noise, explicitly NOT speech."""
    positive_int(sample_rate_hz, "sample_rate_hz")
    positive_int(samples, "samples")
    positive_int(count, "count")
    if samples < sample_rate_hz // 10:
        raise ValueError("Synthetic scenes must be at least 0.1 s long")
    low, high = band_hz
    if not 0 < low < high < sample_rate_hz / 2:
        raise ValueError("Invalid source band")
    rng = np.random.default_rng(seed)
    sos = butter(6, [low, high], btype="bandpass", fs=sample_rate_hz, output="sos")
    time = np.arange(samples) / sample_rate_hz
    sources = []
    for index in range(count):
        source = sosfiltfilt(sos, rng.standard_normal(samples))
        source *= 0.6 + 0.4 * np.sin(2 * np.pi * (2 + index) * time + index) ** 2
        fade = min(samples // 4, int(0.02 * sample_rate_hz))
        source[:fade] *= np.linspace(0, 1, fade)
        source[-fade:] *= np.linspace(1, 0, fade)
        source *= 0.08 / np.sqrt(np.mean(source**2))
        sources.append(source)
    return np.stack(sources)


@dataclass(frozen=True)
class SimulatedScene:
    mixture: NDArray[np.float64]
    components: NDArray[np.float64]
    noise: NDArray[np.float64]
    relative_delays_samples: NDArray[np.float64]


def simulate_far_field(
    sources: ArrayLike,
    array: ArrayConfig,
    angles_deg: ArrayLike,
    *,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> SimulatedScene:
    raw = np.asarray(sources)
    if np.iscomplexobj(raw):
        raise ValueError("Sources must be real")
    data = np.asarray(raw, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError("sources must have shape [source, sample] and be finite/nonempty")
    if angles.shape != (len(data),) or not np.isfinite(angles).all():
        raise ValueError("One finite angle is required per source")
    if np.any(np.abs(angles) > 90):
        raise ValueError("Only front half-plane source directions are supported")
    noise_rms = finite_real(noise_rms, "noise_rms")
    if noise_rms < 0:
        raise ValueError("noise_rms must be nonnegative")
    # Deliberately independent scalar path difference, not steering_vectors or
    # an STFT phase shift: mismatched direction signs cannot cancel by construction.
    components = np.empty((len(data), data.shape[1], array.microphone_count))
    delays = np.empty((len(data), array.microphone_count))
    for source_index, angle in enumerate(angles):
        for mic in range(array.microphone_count):
            path_difference = -(mic - array.reference_microphone_index) * array.spacing_m
            path_difference *= np.sin(float(angle) * np.pi / 180)
            if array.positions_m is not None:
                px, py, _ = array.positions_m[mic]
                rx, ry, _ = array.positions_m[array.reference_microphone_index]
                radians = float(angle) * np.pi / 180
                path_difference = -(px - rx) * np.sin(radians) - (py - ry) * np.cos(radians)
            delay = path_difference / array.sound_speed_m_per_s * array.sample_rate_hz
            delays[source_index, mic] = delay
            components[source_index, :, mic] = fractional_delay(data[source_index], float(delay))
    noise = np.random.default_rng(seed).normal(
        0, noise_rms, size=(data.shape[1], array.microphone_count)
    )
    mixture = components.sum(axis=0) + noise
    return SimulatedScene(mixture, components, noise, delays)
