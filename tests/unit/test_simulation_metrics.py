import numpy as np
import pytest

from acoustic_array.evaluation.metrics import si_sdr
from acoustic_array.simulation.free_field import fractional_delay, simulate_far_field


@pytest.mark.parametrize("delay", [-7, 0, 12])
def test_integer_delay_impulse(delay):
    source = np.zeros(500)
    source[200] = 1
    expected = np.zeros_like(source)
    expected[200 + delay] = 1
    np.testing.assert_allclose(fractional_delay(source, delay), expected, atol=1e-14)


@pytest.mark.parametrize("delay", [-3.7, 0.25, 4.9])
def test_fractional_delay_against_analytic_sinusoid(delay):
    time = np.arange(8192)
    source = np.cos(2 * np.pi * 0.07 * time)
    expected = np.cos(2 * np.pi * 0.07 * (time - delay))
    output = fractional_delay(source, delay)
    assert np.max(np.abs(output[200:-200] - expected[200:-200])) < 2e-5


def test_physical_reference_and_noise_reproducibility(array):
    source = np.random.default_rng(1).normal(size=(1, 1000))
    first = simulate_far_field(source, array, [35], noise_rms=0.01, seed=123)
    second = simulate_far_field(source, array, [35], noise_rms=0.01, seed=123)
    np.testing.assert_allclose(first.components[0, :, 7], source[0], atol=1e-12)
    np.testing.assert_array_equal(first.mixture, second.mixture)
    np.testing.assert_allclose(first.mixture, first.components.sum(axis=0) + first.noise)
    assert first.relative_delays_samples[0, 15] < 0


def test_si_sdr_known_orthogonal_error_and_scale_invariance():
    t = np.arange(1000)
    target = np.sin(2 * np.pi * t / 100)
    noise = np.cos(2 * np.pi * t / 100)
    estimate = 2 * target + 0.2 * noise
    assert si_sdr(target, estimate) == pytest.approx(20, abs=1e-12)
    assert si_sdr(target, estimate * 0.031) == pytest.approx(20, abs=1e-12)
    assert si_sdr(target, np.zeros_like(target)) == -np.inf
    with pytest.raises(ValueError, match="silent"):
        si_sdr(np.zeros(20), np.zeros(20))
