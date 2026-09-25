import numpy as np
import pytest

from acoustic_array.evaluation.room_components import target_component_summary


def test_coherent_cross_power_and_gain():
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    # Reflection cancels half the direct signal: cannot sum direct/reflected powers.
    result = target_component_summary(x, 2 * x, x, 0.1 * x, 0.01 * x, x + 0.1 * x)
    assert result["direct_projection_gain"] == pytest.approx(2)
    assert result["direct_power"] == 4
    assert result["reflection_power"] == 1
    assert result["direct_reflection_cross_power"] == -4
    assert result["target_total_power"] == 1
    assert result["total_target_ratios"]["sir"]["db"] == pytest.approx(20)
    assert result["direct_to_other_sources"]["db"] == pytest.approx(26.020599913)


def test_direct_distortion_detected():
    x = np.array([-1.0, 1.0, -1.0, 1.0])
    orthogonal = np.array([-1.0, -1.0, 1.0, 1.0])
    d = x + 0.5 * orthogonal
    result = target_component_summary(x, d, d, 0.1 * orthogonal, 0.01 * x, x + orthogonal)
    assert result["direct_projection_gain"] == pytest.approx(1)
    assert result["direct_error_to_reference"]["db"] == pytest.approx(-6.020599913)
    assert result["reflection_to_direct"]["status"] == "negative_infinity"


def test_bad_inputs():
    x = np.arange(4.0)
    for bad in [x.astype(complex), x[:2], np.full(4, np.nan)]:
        with pytest.raises(ValueError):
            target_component_summary(x, bad, x, x, x, x)
