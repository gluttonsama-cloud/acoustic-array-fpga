"""Controls for frozen component composition, scoring, and archive seals."""

from pathlib import Path

import numpy as np
import pytest

from acoustic_array.io.artifacts import sha256_file, write_json
from acoustic_array.localization.scan import DirectionPeak
from tools.w4_boundary import check_baseline, compose_subset, score_peaks, verify_run


def test_subset_composition_keeps_full_scene_levels_and_one_fixed_noise() -> None:
    full = np.array([[[2.0], [4.0]], [[3.0], [6.0]], [[5.0], [10.0]]])
    direct = full / 2
    noise = np.array([[0.1], [-0.2]])
    assert np.array_equal(compose_subset(full, direct, noise, (0,), "reverberant"), full[0] + noise)
    assert np.array_equal(
        compose_subset(full, direct, noise, (0, 2), "reverberant"),
        full[0] + full[2] + noise,
    )
    assert np.array_equal(
        compose_subset(full, direct, noise, (1, 2), "direct_only"),
        direct[1] + direct[2] + noise,
    )
    assert np.array_equal(
        compose_subset(full, direct, noise, (0, 1, 2), "reverberant"),
        full.sum(axis=0) + noise,
    )


def test_subset_rejects_invalid_mode_and_source_indices() -> None:
    components = np.zeros((3, 4, 2))
    noise = np.ones((4, 2))
    with pytest.raises(ValueError, match="mode"):
        compose_subset(components, components, noise, (0,), "dry")
    for subset in ((), (0, 0), (3,), (-1,)):
        with pytest.raises(ValueError, match="subset"):
            compose_subset(components, components, noise, subset, "direct_only")


def test_scoring_maximizes_in_gate_matches_and_counts_false_peaks() -> None:
    truth = np.array([-15.0, 0.0, 15.0])
    peaks = (DirectionPeak(3.0, 1.0), DirectionPeak(17.0, 0.8), DirectionPeak(-65.0, 0.4))
    scored = score_peaks(truth, peaks, 5.0)
    assert scored["matched"] == 2
    assert scored["missed"] == 1
    assert scored["false_peaks"] == 1
    assert sorted(scored["matched_errors_deg"]) == [2.0, 3.0]


def test_baseline_replay_checks_peaks_scores_and_match_counts() -> None:
    baseline = {
        "method": "normmusic",
        "frames": 561,
        "peaks": [{"angle_deg": 1.0, "score": 1.0}],
        "matched": 1,
        "missed": 2,
        "false_peaks": 0,
    }
    replay = {**baseline, "source_count_control": 3}
    check_baseline(replay, baseline)
    with pytest.raises(ValueError, match="peaks differ"):
        check_baseline({**replay, "peaks": [{"angle_deg": 2.0, "score": 1.0}]}, baseline)
    with pytest.raises(ValueError, match="matched differs"):
        check_baseline({**replay, "matched": 0}, baseline)


def test_baseline_requires_completion_and_all_output_hashes(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_text("frozen", encoding="utf-8")
    write_json(
        tmp_path / "manifest.json",
        {"outputs_sha256": {"data.txt": sha256_file(tmp_path / "data.txt")}},
    )
    with pytest.raises(ValueError, match="Missing completion"):
        verify_run(tmp_path, require_completion=True)
    evidence = verify_run(tmp_path, require_completion=False)
    assert evidence["completion_present"] is False
    write_json(
        tmp_path / "completion.json", {"manifest_sha256": sha256_file(tmp_path / "manifest.json")}
    )
    assert verify_run(tmp_path, require_completion=True)["completion_present"] is True
    (tmp_path / "data.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid archived output"):
        verify_run(tmp_path, require_completion=True)
