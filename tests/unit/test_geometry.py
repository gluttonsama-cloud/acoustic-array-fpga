import copy

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import microphone_positions, relative_delays, steering_vectors
from acoustic_array.io.artifacts import load_json


def test_geometry_and_independent_endfire_delay(array):
    positions = microphone_positions(array)
    np.testing.assert_allclose(positions[[0, -1], 0], [-0.2286, 0.2286], atol=1e-15)
    delays = relative_delays(array, [-90, 0, 90])
    assert delays[15, 2] == pytest.approx(-8 * 0.03048 / 343)
    assert delays[0, 2] == pytest.approx(7 * 0.03048 / 343)
    np.testing.assert_array_equal(delays[:, 1], 0)
    np.testing.assert_array_equal(delays[7], 0)
    np.testing.assert_allclose(delays[:, 0], -delays[:, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("microphone_count", True),
        ("sample_rate_hz", 0),
        ("spacing_m", float("nan")),
        ("reference_microphone_index", 16),
        ("aperture_m", 1),
        ("pdm_clock_hz", 1),
        ("spatial_processing_band_hz", [0, 24000]),
        ("schema_version", "9"),
    ],
)
def test_bad_config_rejected(project_root, field, value):
    raw = load_json(project_root / "configs/array/ula16_v1.json")
    modified = copy.deepcopy(raw)
    modified[field] = value
    with pytest.raises(ValueError):
        ArrayConfig.from_mapping(modified)


def test_swapped_channel_contract_rejected(project_root):
    raw = load_json(project_root / "configs/array/ula16_v1.json")
    raw["geometry"]["channel_order"] = "reversed"
    with pytest.raises(ValueError, match="channel_order"):
        ArrayConfig.from_mapping(raw)


def test_steering_phase_matches_one_sample_delay():
    array = ArrayConfig(48000, 2, 343 / 48000, 0)
    # +90 degrees reaches MIC1 one sample before MIC0.
    a = steering_vectors(array, [0, 12000], [90])
    assert a[1, 1, 0] == pytest.approx(1j)
    with pytest.raises(ValueError):
        steering_vectors(array, [25000], [0])


@pytest.mark.parametrize("n,h", [(255, 128), (512, 128), (True, 1), (2, 1)])
def test_stft_invalid(n, h):
    with pytest.raises(ValueError):
        STFTConfig(n, h)
