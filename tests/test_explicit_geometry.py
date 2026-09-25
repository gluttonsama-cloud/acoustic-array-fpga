from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.core.geometry import microphone_positions, relative_delays, steering_vectors
from acoustic_array.io.artifacts import load_json
from acoustic_array.simulation.free_field import simulate_far_field
from acoustic_array.simulation.near_field import simulate_near_field


def test_explicit_ula_preserves_reference_and_waveforms():
    array = ArrayConfig(16000, 4, 0.03, 1)
    positions = microphone_positions(array)
    explicit = replace(array, positions_m=positions.tolist())
    np.testing.assert_array_equal(microphone_positions(explicit), positions)
    np.testing.assert_array_equal(
        relative_delays(explicit, [-30, 0, 30]), relative_delays(array, [-30, 0, 30])
    )
    for distance in (None, [2]):
        np.testing.assert_array_equal(
            steering_vectors(explicit, [0, 1000, 4000], [30], source_distances_m=distance),
            steering_vectors(array, [0, 1000, 4000], [30], source_distances_m=distance),
        )
    x = np.random.default_rng(1).normal(size=(1, 512))
    for simulator, args in [(simulate_far_field, ()), (simulate_near_field, ([2],))]:
        a = simulator(x, array, [30], *args)
        b = simulator(x, explicit, [30], *args)
        np.testing.assert_allclose(a.components, b.components, atol=2e-12)
    positions[:] = 100
    assert np.max(np.abs(microphone_positions(explicit))) < 1


def test_three_dimensional_nearfield_and_front_plane_delays():
    array = ArrayConfig(
        16000, 3, 0.03, 0, positions_m=((0, -0.1, -0.1), (0, 0.1, -0.1), (0, 0, 0.2))
    )
    x = np.zeros((1, 256))
    x[0, 100] = 1
    far = simulate_far_field(x, array, [0])
    expected = -np.array([0, 0.2, 0.1]) / 343 * 16000
    np.testing.assert_allclose(far.relative_delays_samples[0], expected)
    np.testing.assert_allclose(relative_delays(array, [0])[:, 0] * 16000, expected)
    near = simulate_near_field(x, array, [0], [2])
    paths = np.sqrt(np.array([2.1**2 + 0.1**2, 1.9**2 + 0.1**2, 2**2 + 0.2**2]))
    np.testing.assert_allclose(near.relative_delays_samples[0], (paths - paths[0]) / 343 * 16000)


@pytest.mark.parametrize(
    "positions",
    [
        [[0, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [2, 0, 0]],
        [[True, 0, 0], [-1, 0, 0]],
        [[float("nan"), 0, 0], [0, 0, 0]],
        [[1j, 0, 0], [-1j, 0, 0]],
    ],
)
def test_invalid_coordinates(positions):
    with pytest.raises(ValueError):
        ArrayConfig(16000, 2, 0.03, 0, positions_m=positions)


def test_mapping_schema_is_explicit_and_aperture_checked():
    raw = load_json(Path(__file__).resolve().parents[1] / "configs/array/ula16_v1.json")
    array = ArrayConfig.from_mapping(raw)
    raw["positions_m"] = microphone_positions(array).tolist()
    with pytest.raises(ValueError, match="schema_version"):
        ArrayConfig.from_mapping(raw)
    raw["schema_version"] = "0.2"
    raw["geometry"].update(
        kind="explicit_coordinates", axis="xyz", channel_order="MIC0_to_MIC15_row_order"
    )
    with pytest.raises(ValueError, match="coordinate_formula"):
        ArrayConfig.from_mapping(raw)
    raw["geometry"].pop("coordinate_formula")
    parsed = ArrayConfig.from_mapping(raw)
    np.testing.assert_array_equal(microphone_positions(parsed), microphone_positions(array))
    raw["aperture_m"] *= 2
    with pytest.raises(ValueError, match="aperture"):
        ArrayConfig.from_mapping(raw)
