import numpy as np
import pytest

from acoustic_array.simulation.weak_scene import calibrate_weak_scene


def test_target_mask_controls_ratio_and_geometry_is_preserved():
    target = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, -2.0], [0.0, 0.0]])
    other = np.array([[100.0, 200.0], [3.0, 6.0], [-3.0, -6.0], [100.0, 200.0]])
    mask = np.array([False, True, True, False])
    original = other.copy()
    result = calibrate_weak_scene(
        target, other, mask, reference_microphone=1, target_rms=0.01, sir_db=-20
    )
    # The interferer is 10x target amplitude on target events; off-event energy
    # must not alter the calibration. One scalar preserves inter-mic responses.
    np.testing.assert_allclose(result.target[mask, 1], [0.01, -0.01])
    np.testing.assert_allclose(result.interference[mask, 1], [0.1, -0.1])
    np.testing.assert_allclose(result.interference[:, 1], 2 * result.interference[:, 0])
    assert result.measured_sir_db == pytest.approx(-20)
    assert result.interference[0, 1] > 1  # do not silently clip
    np.testing.assert_array_equal(other, original)


@pytest.mark.parametrize("kind", ["zero", "nan", "complex", "mask", "reference"])
def test_invalid_calibration(kind):
    x = np.ones((4, 2))
    y = x.copy()
    mask = np.ones(4, bool)
    ref = 0
    if kind == "zero":
        y[:] = 0
    if kind == "nan":
        y[0, 0] = np.nan
    if kind == "complex":
        y = y.astype(complex)
    if kind == "mask":
        mask = mask.astype(int)
    if kind == "reference":
        ref = True
    with pytest.raises(ValueError):
        calibrate_weak_scene(x, y, mask, reference_microphone=ref, target_rms=0.01, sir_db=-10)
