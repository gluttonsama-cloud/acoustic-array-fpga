import numpy as np
import pytest

from acoustic_array.beamforming.das import das_weights
from acoustic_array.beamforming.sector_lcmv import sector_lcmv_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import steering_vectors

ARRAY = ArrayConfig(16000, 16, 0.03, 1)
STFT = STFTConfig(512, 256)
FREQUENCIES = np.fft.rfftfreq(STFT.fft_size, 1 / ARRAY.sample_rate_hz)
IDENTITY = np.broadcast_to(np.eye(16), (len(FREQUENCIES), 16, 16)).copy()


def test_identity_covariance_matches_independent_minimum_norm_solution():
    result = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20], min_white_noise_gain_db=-100)
    f = 32  # 1000 Hz: feasible, well-conditioned three-point constraints.
    assert result.status[f, 0] == "sector_lcmv"
    c = steering_vectors(ARRAY, [FREQUENCIES[f]], [18, 20, 22])[0]
    expected = np.linalg.lstsq(c.conj().T, np.ones(3), rcond=None)[0]
    np.testing.assert_allclose(result.weights[f, :, 0], expected, atol=1e-9)
    np.testing.assert_allclose(c.conj().T @ result.weights[f, :, 0], 1, atol=1e-9)
    assert result.constraint_residual[f, 0] < 1e-9
    assert result.condition_number[f] >= 1


def test_shape_independent_targets_and_real_endpoints():
    result = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [-20, 20], min_white_noise_gain_db=-100)
    assert result.weights.shape == (257, 16, 2)
    assert result.status.shape == (257, 2)
    assert result.condition_number.shape == (257,)
    assert result.constraint_residual.shape == (257, 2)
    assert result.white_noise_gain_db.shape == (257, 2)
    single = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [-20], min_white_noise_gain_db=-100)
    np.testing.assert_allclose(result.weights[:, :, 0], single.weights[:, :, 0])
    assert np.isrealobj(result.weights[0].real)
    np.testing.assert_array_equal(result.weights[0].imag, 0)
    np.testing.assert_array_equal(result.weights[-1].imag, 0)
    das = das_weights(ARRAY, STFT, [-20, 20])
    np.testing.assert_array_equal(result.weights[0], das[0])
    np.testing.assert_array_equal(result.weights[-1], das[-1])
    assert np.all(result.status[[0, -1]] == "outside_band_das")
    assert np.all(np.isnan(result.constraint_residual[[0, -1]]))


def test_fallbacks_report_three_point_residual():
    das = das_weights(ARRAY, STFT, [20])
    result = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20], min_white_noise_gain_db=100)
    np.testing.assert_array_equal(result.weights, das)
    assert result.status[32, 0] == "fallback_white_noise_gain"
    c = steering_vectors(ARRAY, [1000], [18, 20, 22])[0]
    np.testing.assert_allclose(
        result.constraint_residual[32, 0],
        np.max(np.abs(c.conj().T @ das[32, :, 0] - 1)),
    )
    zero = sector_lcmv_weights(IDENTITY * 0, ARRAY, STFT, [20])
    np.testing.assert_array_equal(zero.weights, das)
    assert zero.status[32, 0] == "fallback_zero_power"
    ill = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20], max_condition=1)
    assert ill.status[32, 0] == "fallback_rank_or_condition"


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan, True])
def test_invalid_width(width):
    with pytest.raises(ValueError):
        sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20], half_width_deg=width)


@pytest.mark.parametrize("kind", ["nonhermitian", "negative", "nan", "shape"])
def test_invalid_covariance(kind):
    r = IDENTITY.astype(complex).copy()
    if kind == "nonhermitian":
        r[1, 0, 1] = 1j
    elif kind == "negative":
        r[1, 0, 0] = -1
    elif kind == "nan":
        r[1, 0, 0] = np.nan
    else:
        r = r[:10]
    with pytest.raises(ValueError):
        sector_lcmv_weights(r, ARRAY, STFT, [20])


def test_window_and_band_validation():
    with pytest.raises(ValueError):
        sector_lcmv_weights(IDENTITY, ARRAY, STFT, [89])
    with pytest.raises(ValueError):
        sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20], band_hz=(101, 102))
    for angles in ([True], [20 + 1j]):
        with pytest.raises(ValueError):
            sector_lcmv_weights(IDENTITY, ARRAY, STFT, angles)


@pytest.mark.parametrize("scale", [1e-300, 1e308])
def test_extreme_finite_covariance_scale_invariance(scale):
    base = sector_lcmv_weights(IDENTITY, ARRAY, STFT, [20])
    scaled = sector_lcmv_weights(IDENTITY * scale, ARRAY, STFT, [20])
    np.testing.assert_allclose(scaled.weights, base.weights, atol=1e-12)
    np.testing.assert_array_equal(scaled.status, base.status)
