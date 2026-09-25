import numpy as np
import pytest

from acoustic_array.simulation.active_scene import build_active_scene


def signals():
    t = np.arange(12000) / 48000
    x = np.stack([np.sin(2 * np.pi * f * t) for f in [300, 700, 1200]])
    x[0, :3000] = 0
    x[1, 8000:] = 0
    return x


def test_active_levels_noise_and_reference(array):
    result = build_active_scene(signals(), array, [-45, 0, 50], trim_samples=500, seed=19)
    np.testing.assert_allclose(result.active_rms, 0.04, rtol=1e-13)
    assert result.reference_noise_rms == pytest.approx(0.004, rel=1e-13)
    assert not result.masks[:, :500].any() and not result.masks[:, -500:].any()
    np.testing.assert_allclose(
        result.scene.mixture, result.scene.components.sum(axis=0) + result.scene.noise
    )
    np.testing.assert_array_equal(
        result.scene.noise,
        build_active_scene(signals(), array, [-45, 0, 50], trim_samples=500, seed=19).scene.noise,
    )


def test_zero_trim_and_different_duty_cycles(array):
    result = build_active_scene(signals(), array, [-45, 0, 50], trim_samples=0)
    assert result.masks.sum(axis=1)[0] < result.masks.sum(axis=1)[2]
    np.testing.assert_allclose(result.active_rms, 0.04, atol=1e-14)


@pytest.mark.parametrize(
    "kwargs", [{"trim_samples": 6000}, {"seed": True}, {"active_rms": 0}, {"noise_snr_db": np.nan}]
)
def test_invalid_configuration(array, kwargs):
    with pytest.raises(ValueError):
        build_active_scene(signals(), array, [-45, 0, 50], **kwargs)


def test_silence_rejected(array):
    with pytest.raises(ValueError, match="activity"):
        build_active_scene(np.zeros((3, 12000)), array, [-45, 0, 50])
