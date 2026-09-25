import numpy as np
import pytest

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import STFTConfig
from acoustic_array.pipelines.streaming import StreamingBeamformer


def test_independent_analytic_constraints(array):
    stft = STFTConfig(512, 256)
    angles = [-45, 0, 50]
    result = constrained_weights(array, stft, angles)
    for f in [10, 20, 40]:
        frequency = f * 48000 / 512
        c = np.array(
            [
                [
                    np.exp(
                        2j * np.pi * frequency * (m - 7) * 0.03048 * np.sin(np.deg2rad(theta)) / 343
                    )
                    for theta in angles
                ]
                for m in range(16)
            ]
        )
        np.testing.assert_allclose(c.conj().T @ result.weights[f], np.eye(3), atol=1e-12)
        assert np.all(result.status[f] == "constrained")
    assert np.nanmax(result.constraint_residual) < 1e-12
    assert np.nanmin(result.white_noise_gain_db) > 0


def test_band_outside_is_exact_das(array):
    config = STFTConfig(256, 128)
    result = constrained_weights(array, config, [-45, 0, 50])
    baseline = das_weights(array, config, [-45, 0, 50])
    outside = result.status[:, 0] == "outside_band_das"
    np.testing.assert_array_equal(result.weights[outside], baseline[outside])
    assert np.all(np.isreal(result.weights[[0, -1]]))


def test_duplicate_directions_report_fallback(array):
    config = STFTConfig(512, 256)
    result = constrained_weights(array, config, [0, 0, 0])
    active = result.status != "outside_band_das"
    assert np.all(result.status[active] == "fallback_rank_or_condition")
    np.testing.assert_array_equal(result.weights, das_weights(array, config, [0, 0, 0]))
    assert np.nanmax(result.constraint_residual) >= 1 - 1e-12


def test_white_noise_gain_protection(array):
    config = STFTConfig(512, 256)
    result = constrained_weights(
        array, config, [-0.5, 0, 0.5], max_condition=1e12, min_white_noise_gain_db=0
    )
    assert np.any(result.status == "fallback_white_noise_gain")
    assert np.isfinite(result.weights).all()


def test_generic_stream_rejects_bad_endpoints(array):
    weights = das_weights(array, STFTConfig(256, 128), [0])
    weights[-1] += 1j
    with pytest.raises(ValueError, match="Nyquist"):
        StreamingBeamformer(array, STFTConfig(256, 128), weights)


def test_empty_constraint_band_rejected(array):
    with pytest.raises(ValueError, match="no FFT bins"):
        constrained_weights(array, STFTConfig(256, 128), [-45, 0, 50], band_hz=(501, 502))
