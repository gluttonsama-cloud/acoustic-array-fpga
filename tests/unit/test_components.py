import numpy as np
import pytest

from acoustic_array.evaluation.components import (
    component_ratios,
    energy_ratio_db,
    spectral_power_partition,
)


def test_physical_ratios_known_amplitudes():
    x = np.ones(20)
    result = component_ratios(2 * x, 0.2 * x, 0.02 * x)
    assert result["sir"]["db"] == pytest.approx(20)
    assert result["snr"]["db"] == pytest.approx(40)


def test_coherent_interferers_must_be_summed():
    x = np.ones(20)
    result = component_ratios(x, x + (-x), x)
    assert result["sir"]["status"] == "positive_infinity"
    assert result["snr"]["db"] == 0


@pytest.mark.parametrize(
    "a,b,status", [(0, 0, "undefined"), (0, 1, "negative_infinity"), (1, 0, "positive_infinity")]
)
def test_degenerate_ratios(a, b, status):
    assert energy_ratio_db(a, b) == {"db": None, "status": status}


@pytest.mark.parametrize("length", [48000, 48001])
def test_partition_parseval_including_dc_nyquist(length):
    x = np.random.default_rng(4).normal(size=length) + 2
    assert sum(spectral_power_partition(x, 48000).values()) == pytest.approx(
        np.mean(x * x), rel=1e-12
    )


def test_partition_known_tones():
    t = np.arange(48000) / 48000
    x = (
        2 * np.cos(2 * np.pi * 100 * t)
        + 3 * np.cos(2 * np.pi * 1000 * t)
        + 4 * np.cos(2 * np.pi * 8000 * t)
    )
    result = spectral_power_partition(x, 48000)
    assert list(result.values()) == pytest.approx([2, 4.5, 8], rel=1e-12)


@pytest.mark.parametrize("signal", [[], [np.nan], [[1]], [1e308]])
def test_bad_components(signal):
    with pytest.raises(ValueError):
        component_ratios(signal, signal, signal)


def test_mismatched_components():
    with pytest.raises(ValueError):
        component_ratios([1, 2], [1], [1, 2])
