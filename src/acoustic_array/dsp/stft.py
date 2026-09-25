"""Causal framing and normalized weighted overlap-add.

Analysis starts at -N/2 with known zero history. A frame is emitted only
after its entire non-padding input has arrived. Synthesis releases the
first hop of each accumulated frame, which no later frame can modify.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal.windows import hann

from acoustic_array.core.config import STFTConfig, positive_int


def pcm_array(value: ArrayLike, channels: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("PCM must be real")
    data = np.asarray(raw, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != channels or not np.isfinite(data).all():
        raise ValueError(f"Expected finite PCM [sample, {channels}]")
    return data


@dataclass(frozen=True)
class SpectralFrame:
    index: int
    sample_start: int
    spectrum: NDArray[np.complex128]


class StreamingSTFT:
    def __init__(self, config: STFTConfig, channels: int):
        self.config = config
        self.channels = positive_int(channels, "channels")
        self.window = hann(config.fft_size, sym=False)
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros((self.config.left_padding, self.channels))
        self.total_samples = 0
        self._next_start = -self.config.left_padding
        self._frame_index = 0
        self.closed = False

    def _take_frame(self) -> SpectralFrame:
        spectrum = np.fft.rfft(self._buffer[: self.config.fft_size] * self.window[:, None], axis=0)
        frame = SpectralFrame(self._frame_index, self._next_start, spectrum)
        self._buffer = self._buffer[self.config.hop_size :].copy()
        self._next_start += self.config.hop_size
        self._frame_index += 1
        return frame

    def push(self, pcm: ArrayLike, *, sample_start: int) -> list[SpectralFrame]:
        if self.closed:
            raise RuntimeError("STFT is closed; call reset before reusing")
        if isinstance(sample_start, bool) or not isinstance(sample_start, int):
            raise ValueError("sample_start must be an integer")
        if sample_start != self.total_samples:
            raise ValueError("Non-contiguous PCM block: dropped, reordered or repeated samples")
        data = pcm_array(pcm, self.channels)
        self._buffer = np.concatenate((self._buffer, data))
        self.total_samples += len(data)
        frames = []
        while len(self._buffer) >= self.config.fft_size:
            frames.append(self._take_frame())
        return frames

    def flush(self) -> list[SpectralFrame]:
        if self.closed:
            raise RuntimeError("STFT already flushed")
        frames = []
        if self.total_samples:
            while self._next_start < self.total_samples:
                missing = self.config.fft_size - len(self._buffer)
                self._buffer = np.pad(self._buffer, ((0, missing), (0, 0)))
                frames.append(self._take_frame())
        self.closed = True
        return frames


class StreamingISTFT:
    def __init__(self, config: STFTConfig, channels: int):
        self.config = config
        self.channels = positive_int(channels, "channels")
        self.window = hann(config.fft_size, sym=False)
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros((self.config.fft_size, self.channels))
        self._normalization = np.zeros(self.config.fft_size)
        self._next_index = 0
        self._next_start = -self.config.left_padding
        self._emitted_samples = 0
        self.closed = False

    def push(self, frame: SpectralFrame, *, valid_stop: int | None = None) -> NDArray[np.float64]:
        if self.closed:
            raise RuntimeError("ISTFT is closed; call reset before reusing")
        if frame.index != self._next_index or frame.sample_start != self._next_start:
            raise ValueError("Non-contiguous spectral frame")
        spectrum = np.asarray(frame.spectrum)
        expected = (self.config.fft_size // 2 + 1, self.channels)
        if spectrum.shape != expected or not np.isfinite(spectrum).all():
            raise ValueError(f"Expected finite spectrum {expected}")
        if np.max(np.abs(spectrum[[0, -1]].imag)) > 1e-12:
            raise ValueError("DC and Nyquist bins must be real for a real-valued signal")
        if valid_stop is not None and (
            isinstance(valid_stop, bool) or not isinstance(valid_stop, int) or valid_stop < 0
        ):
            raise ValueError("valid_stop must be a nonnegative integer")
        self._sum += np.fft.irfft(spectrum, n=self.config.fft_size, axis=0) * self.window[:, None]
        self._normalization += self.window**2
        hop = self.config.hop_size
        left = max(0, -frame.sample_start)
        right = hop if valid_stop is None else min(hop, max(0, valid_stop - frame.sample_start))
        if right > left:
            norm = self._normalization[left:right]
            if np.any(norm <= np.finfo(float).eps):
                raise ValueError("Invalid overlap-add normalization")
            output = self._sum[left:right] / norm[:, None]
        else:
            output = np.empty((0, self.channels))
        self._sum = np.concatenate((self._sum[hop:], np.zeros((hop, self.channels))))
        self._normalization = np.concatenate((self._normalization[hop:], np.zeros(hop)))
        self._next_index += 1
        self._next_start += hop
        self._emitted_samples += len(output)
        return output

    def finish(self, total_samples: int) -> None:
        if self.closed:
            raise RuntimeError("ISTFT already finished")
        if (
            isinstance(total_samples, bool)
            or not isinstance(total_samples, int)
            or total_samples < 0
        ):
            raise ValueError("total_samples must be a nonnegative integer")
        if total_samples and self._next_start < total_samples:
            raise ValueError("Missing synthesis frames; flush the analyzer first")
        if self._emitted_samples != total_samples:
            raise ValueError(
                "Emitted sample count does not match total_samples; check tail cropping"
            )
        self.closed = True
