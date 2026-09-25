"""Static known-direction beamformer; no automatic DOA or hardware claims."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acoustic_array.beamforming.das import apply_weights, das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, positive_int
from acoustic_array.dsp.stft import SpectralFrame, StreamingISTFT, StreamingSTFT, pcm_array


class StreamingBeamformer:
    def __init__(self, array: ArrayConfig, stft: STFTConfig, weights: ArrayLike):
        self.weights = np.array(weights, dtype=np.complex128, copy=True)
        expected = (stft.fft_size // 2 + 1, array.microphone_count)
        if (
            self.weights.ndim != 3
            or self.weights.shape[:2] != expected
            or not self.weights.shape[2]
            or not np.isfinite(self.weights).all()
        ):
            raise ValueError("Weights must be finite [frequency,microphone,output]")
        if np.any(np.abs(self.weights[[0, -1]].imag) > 1e-12):
            raise ValueError("DC/Nyquist weights must be real")
        self.weights.flags.writeable = False
        self.analysis = StreamingSTFT(stft, array.microphone_count)
        self.synthesis = StreamingISTFT(stft, self.weights.shape[2])

    def reset(self) -> None:
        self.analysis.reset()
        self.synthesis.reset()

    def _process(
        self, frames: list[SpectralFrame], valid_stop: int | None = None
    ) -> NDArray[np.float64]:
        outputs = []
        for frame in frames:
            transformed = SpectralFrame(
                frame.index, frame.sample_start, apply_weights(frame.spectrum, self.weights)
            )
            outputs.append(self.synthesis.push(transformed, valid_stop=valid_stop))
        return np.concatenate(outputs) if outputs else np.empty((0, self.weights.shape[2]))

    def push(self, pcm: ArrayLike, *, sample_start: int) -> NDArray[np.float64]:
        return self._process(self.analysis.push(pcm, sample_start=sample_start))

    def flush(self) -> NDArray[np.float64]:
        output = self._process(self.analysis.flush(), self.analysis.total_samples)
        self.synthesis.finish(self.analysis.total_samples)
        return output


class StreamingDAS(StreamingBeamformer):
    def __init__(self, array: ArrayConfig, stft: STFTConfig, angles_deg: ArrayLike):
        super().__init__(array, stft, das_weights(array, stft, angles_deg))


def beamform_with_weights(
    pcm: ArrayLike,
    array: ArrayConfig,
    stft: STFTConfig,
    weights: ArrayLike,
    *,
    block_size: int = 1024,
) -> NDArray[np.float64]:
    positive_int(block_size, "block_size")
    data = pcm_array(pcm, array.microphone_count)
    processor = StreamingBeamformer(array, stft, weights)
    blocks = [
        processor.push(data[i : i + block_size], sample_start=i)
        for i in range(0, len(data), block_size)
    ]
    blocks.append(processor.flush())
    return np.concatenate(blocks)


def beamform_pcm(
    pcm: ArrayLike,
    array: ArrayConfig,
    stft: STFTConfig,
    angles_deg: ArrayLike,
    *,
    block_size: int = 1024,
) -> NDArray[np.float64]:
    return beamform_with_weights(
        pcm, array, stft, das_weights(array, stft, angles_deg), block_size=block_size
    )
