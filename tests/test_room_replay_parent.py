import pytest

from acoustic_array.pipelines.room_component_replay import frozen_weight_path, parent_methods


@pytest.mark.parametrize("loadings", [[0.1], [0.1, 1.0], [0.1, 1.0, 2.0]])
def test_parent_matrix_is_complete_and_unique(loadings):
    parent = {"config": {"diagonal_loadings": loadings}, "selected_cases": ["a", "b"]}
    names = ["das", "fixed_constraint"] + [f"mpdr_load{v}" for v in loadings]
    rows = [{"case": case, "method": method} for case in ["a", "b"] for method in names]
    prior = {"expected_cases": 2, "completed_cases": 2, "rows": rows}
    assert parent_methods(parent, prior) == set(names)
    for bad in [rows[:-1], rows[:-1] + [rows[0]], rows + [rows[0]]]:
        with pytest.raises(ValueError, match="matrix"):
            parent_methods(parent, {**prior, "rows": bad})


def test_mismatched_baseline_never_falls_back_to_oracle(tmp_path):
    row = {"case": "r2_gap15_rt0", "distance_m": 2, "adjacent_angle_deg": 15}
    room = tmp_path / "room"
    room.mkdir()
    oracle = room / "r2_gap15_512_weights.npz"
    oracle.touch()
    parent = {"config": {"fft_size": 512}}
    assert frozen_weight_path(tmp_path, room, parent, row, "das") == (oracle, "das")
    parent["config"]["steering_error"] = {"angle_offset_deg": 2, "distance_offset_m": 0}
    with pytest.raises(ValueError, match="Missing geometry"):
        frozen_weight_path(tmp_path, room, parent, row, "fixed_constraint")
    changed = tmp_path / "r2_gap15_rt0_baseline_weights.npz"
    changed.touch()
    assert frozen_weight_path(tmp_path, room, parent, row, "fixed_constraint") == (
        changed,
        "constrained",
    )
    assert frozen_weight_path(tmp_path, room, parent, row, "mpdr_load0.1") == (
        tmp_path / "r2_gap15_rt0_mpdr_load0.1_weights.npz",
        "weights",
    )
