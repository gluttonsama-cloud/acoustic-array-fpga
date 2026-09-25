import numpy as np
import pytest

from acoustic_array.core.config import STFTConfig
from acoustic_array.evaluation.metrics import si_sdr
from acoustic_array.pipelines.streaming import StreamingDAS, beamform_pcm
from acoustic_array.simulation.free_field import simulate_far_field, synthetic_sources


def test_broadside_identity_including_dc_nyquist(array):
    signal = np.random.default_rng(7).normal(size=3073)
    signal += 0.3 + 0.5 * (-1.0) ** np.arange(len(signal))
    pcm = np.repeat(signal[:, None], array.microphone_count, axis=1)
    output = beamform_pcm(pcm, array, STFTConfig(512, 256), [0], block_size=113)
    np.testing.assert_allclose(output[:, 0], signal, rtol=1e-12, atol=1e-12)


def test_direction_sign_gain_and_channel_reversal(array):
    fs = array.sample_rate_hz
    frequency = 3000
    t = np.arange(fs // 4) / fs
    theta = np.deg2rad(40)
    # Independent analytic plane wave. No simulation or steering helper used here.
    pcm = np.column_stack(
        [
            np.cos(2 * np.pi * frequency * (t + (m - 7) * 0.03048 * np.sin(theta) / 343))
            for m in range(16)
        ]
    )
    angles = [-40, 0, 40]
    output = beamform_pcm(pcm, array, STFTConfig(512, 256), angles, block_size=257)
    power = np.mean(output[1024:-1024] ** 2, axis=0)
    assert np.argmax(power) == 2
    assert power[2] > 20 * power[0]
    gain = 2 * np.mean(output[1024:-1024, 2] * np.cos(2 * np.pi * frequency * t[1024:-1024]))
    assert 0.9 < gain < 1.02  # Finite-window subband DAS has known amplitude error.
    reversed_out = beamform_pcm(pcm[:, ::-1], array, STFTConfig(512, 256), angles)
    assert np.argmax(np.mean(reversed_out[1024:-1024] ** 2, axis=0)) == 0


@pytest.mark.parametrize("count", [1, 2, 3])
def test_synthetic_source_ladder_and_chunk_independence(array, count):
    angles = [-45, 0, 50][:count]
    sources = synthetic_sources(48000, 12000, count, 44, (500, 5000))
    scene = simulate_far_field(sources, array, angles, noise_rms=0.002, seed=11)
    a = beamform_pcm(scene.mixture, array, STFTConfig(512, 256), angles, block_size=113)
    b = beamform_pcm(scene.mixture, array, STFTConfig(512, 256), angles, block_size=12000)
    np.testing.assert_array_equal(a, b)
    for k in range(count):
        reference = scene.components[k, 1024:-1024, 7]
        result = si_sdr(reference, a[1024:-1024, k])
        if count > 1:
            baseline = si_sdr(reference, scene.mixture[1024:-1024, 7])
            assert result - baseline > 6
        else:
            assert result > 20


def test_empty_stream_and_lifecycle(array):
    processor = StreamingDAS(array, STFTConfig(256, 128), [0])
    assert processor.flush().shape == (0, 1)
    with pytest.raises(RuntimeError):
        processor.flush()
    processor.reset()
    processor.push(np.zeros((5, 16)), sample_start=0)
    np.testing.assert_array_equal(processor.flush(), np.zeros((5, 1)))


def test_online_emission_and_future_independence(array):
    pcm = np.random.default_rng(73).normal(size=(2048, 16))
    first = StreamingDAS(array, STFTConfig(512, 256), [0])
    second = StreamingDAS(array, STFTConfig(512, 256), [0])
    assert first.push(pcm[:256], sample_start=0).shape == (0, 1)
    prefix_a = first.push(pcm[256:512], sample_start=256)
    assert prefix_a.shape == (256, 1)
    prefix_b = second.push(pcm[:512], sample_start=0)
    np.testing.assert_array_equal(prefix_a, prefix_b)
    snapshot = prefix_a.copy()
    first.push(pcm[512:], sample_start=512)
    second.push(-pcm[512:], sample_start=512)
    np.testing.assert_array_equal(prefix_a, snapshot)
