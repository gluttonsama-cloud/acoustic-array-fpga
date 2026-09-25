import numpy as np
import pytest

from acoustic_array.simulation.mismatch import apply_channel_mismatch


def test_identity_is_exact_and_nonmutating():
    x = np.random.default_rng(3).normal(size=(3, 1000, 4))
    y = apply_channel_mismatch(x, np.zeros(4), np.zeros(4))
    np.testing.assert_array_equal(x, y)
    assert not np.shares_memory(x, y)


def test_gain_and_integer_impulse_delay():
    x = np.zeros((100, 2))
    x[40] = 1
    y = apply_channel_mismatch(x, [20, -20], [2, -1])
    expected = np.zeros_like(x)
    expected[42, 0] = 10
    expected[39, 1] = 0.1
    np.testing.assert_allclose(y, expected, atol=1e-13)


def test_fractional_delay_against_analytic_tone():
    t = np.arange(4000)
    x = np.cos(2 * np.pi * 0.04 * t)[:, None]
    y = apply_channel_mismatch(x, [0], [0.5])[:, 0]
    np.testing.assert_allclose(
        y[100:-100], np.cos(2 * np.pi * 0.04 * (t[100:-100] - 0.5)), atol=2e-5
    )


def test_linear_component_sum():
    x = np.random.default_rng(3).normal(size=(3, 1000, 4))
    gains = [-1, 0, 1, -0.5]
    delays = [0.5, 0, -0.5, 0.1]
    np.testing.assert_allclose(
        apply_channel_mismatch(x, gains, delays).sum(axis=0),
        apply_channel_mismatch(x.sum(axis=0), gains, delays),
        atol=3e-15,
    )


@pytest.mark.parametrize(
    "g,d", [([1], [0, 0]), ([np.nan, 0], [0, 0]), ([0, 0], [5, 0]), ([21, 0], [0, 0])]
)
def test_invalid_parameters(g, d):
    with pytest.raises(ValueError):
        apply_channel_mismatch(np.ones((10, 2)), g, d)
