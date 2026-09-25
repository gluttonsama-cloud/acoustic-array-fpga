import numpy as np
import pytest
from scipy.signal import stft as scipy_stft

from acoustic_array.core.config import STFTConfig
from acoustic_array.dsp.stft import SpectralFrame, StreamingISTFT, StreamingSTFT


def roundtrip(x, n, block):
    config = STFTConfig(n, n // 2)
    analysis = StreamingSTFT(config, x.shape[1])
    synthesis = StreamingISTFT(config, x.shape[1])
    frames = []
    outputs = []
    for start in range(0, len(x), block):
        new = analysis.push(x[start : start + block], sample_start=start)
        frames.extend(new)
        outputs.extend(synthesis.push(frame) for frame in new)
    tail = analysis.flush()
    frames.extend(tail)
    outputs.extend(synthesis.push(frame, valid_stop=len(x)) for frame in tail)
    synthesis.finish(len(x))
    y = np.concatenate(outputs) if outputs else np.empty_like(x)
    return y, frames


@pytest.mark.parametrize("n", [256, 512])
@pytest.mark.parametrize("length", [0, 1, 17, 128, 256, 512, 1025, 4073])
@pytest.mark.parametrize("block", [1, 73, 1024])
def test_reconstruction_arbitrary_chunks(n, length, block):
    x = np.random.default_rng(71).normal(size=(length, 3))
    y, _ = roundtrip(x, n, block)
    assert y.shape == x.shape
    np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["dc", "nyquist", "impulse", "silence"])
def test_special_signals(kind):
    x = np.zeros((1001, 1))
    if kind == "dc":
        x[:] = 0.2
    elif kind == "nyquist":
        x[:, 0] = (-1.0) ** np.arange(len(x))
    elif kind == "impulse":
        x[[0, 255, 256, -1], 0] = 1
    y, _ = roundtrip(x, 512, 137)
    np.testing.assert_allclose(y, x, atol=1e-12)


def test_cross_reference_scipy():
    x = np.random.default_rng(1).normal(size=(3000, 2))
    _, frames = roundtrip(x, 256, 171)
    _, _, expected = scipy_stft(
        x, fs=48000, window="hann", nperseg=256, noverlap=128, boundary="zeros", padded=True, axis=0
    )
    # scipy.signal.stft scales by window.sum() and returns [frequency,channel,frame].
    expected = np.moveaxis(expected, -1, 0) * 128
    actual = np.stack([f.spectrum for f in frames])
    np.testing.assert_allclose(actual, expected[: len(actual)], atol=1e-12)


def test_no_future_data_and_reset():
    analysis = StreamingSTFT(STFTConfig(256, 128), 1)
    assert analysis.push(np.ones((127, 1)), sample_start=0) == []
    frames = analysis.push(np.ones((1, 1)), sample_start=127)
    assert len(frames) == 1 and frames[0].sample_start == -128
    with pytest.raises(ValueError, match="Non-contiguous"):
        analysis.push(np.ones((1, 1)), sample_start=129)
    analysis.flush()
    with pytest.raises(RuntimeError):
        analysis.flush()
    with pytest.raises(RuntimeError):
        analysis.push(np.ones((1, 1)), sample_start=128)
    analysis.reset()
    assert analysis.total_samples == 0
    assert len(analysis.push(np.ones((128, 1)), sample_start=0)) == 1


def test_reject_corrupt_spectra():
    synthesis = StreamingISTFT(STFTConfig(256, 128), 1)
    spectrum = np.ones((129, 1), dtype=complex)
    with pytest.raises(ValueError, match="Non-contiguous"):
        synthesis.push(SpectralFrame(1, -128, spectrum))
    spectrum[-1] += 1j
    with pytest.raises(ValueError, match="Nyquist"):
        synthesis.push(SpectralFrame(0, -128, spectrum))
    with pytest.raises(ValueError, match="Missing"):
        synthesis.finish(256)


def test_finish_detects_incorrect_tail_cropping():
    config = STFTConfig(256, 128)
    analysis = StreamingSTFT(config, 1)
    synthesis = StreamingISTFT(config, 1)
    frames = analysis.push(np.ones((200, 1)), sample_start=0) + analysis.flush()
    for frame in frames:
        synthesis.push(frame, valid_stop=150)
    with pytest.raises(ValueError, match="Emitted sample count"):
        synthesis.finish(200)
