"""Independent protocol checks for the frozen W4 audio comparison."""

from types import SimpleNamespace

import numpy as np
import pytest

from tools import w4_audio


def test_angular_tie_preserves_correct_ports_and_failed_target():
    # Two assignments have the same total error of 67 degrees. One preserves
    # the zero-degree target and 15-degree target as genuinely correct matches.
    links = w4_audio.match_ports(np.array([-15.0, 0.0, 15.0]), np.array([1.0, 15.0, 51.0]), 5.0)
    assert [link["port"] for link in links] == [2, 0, 1]
    assert [link["localized_within_tolerance"] for link in links] == [False, True, True]
    assert len(links) == 3
    # Scoring follows that permutation, even when a wrong direction would sound better.
    output = np.array([[11.0, 22.0, 33.0]])
    assert output[:, [link["port"] for link in links]].tolist() == [[33.0, 11.0, 22.0]]


def _row(case, branch, method, values):
    result = {"case": case, "branch": branch, "method": method}
    for reference in ("direct_reference", "reverberant_reference"):
        result[reference] = {}
        for band in ("fullband", "evaluation_band"):
            result[reference][band] = {
                "targets": [
                    {
                        "status": "ok",
                        "active_samples": 1000,
                        "metrics": {
                            "input_si_sdr_db": 0.0,
                            "output_si_sdr_db": value,
                            "si_sdri_db": value,
                        }
                        if value is not None
                        else None,
                    }
                    for value in values
                ]
            }
    return result


def _matrix(baseline, onset):
    rows = []
    for case, b, o in zip(w4_audio.CASES, baseline, onset, strict=True):
        for branch, minimum in (("baseline", b), ("onset", o), ("oracle", 8.0)):
            for method in w4_audio.METHODS:
                rows.append(_row(case, branch, method, [minimum + 2.0, minimum, minimum + 1.0]))
    return rows


def test_gate_uses_difference_of_medians_and_per_case_regression():
    cfg = {"minimum_median_gain_db": 1.0, "maximum_case_regression_db": 1.0}
    # Median of paired gains is 2, but the actual median of minima differs by 0.5.
    result = w4_audio.decision_gate(
        _matrix([0.0, 10.0, 11.0, 20.0], [2.0, 9.0, 13.0, 22.0]), w4_audio.CASES, cfg
    )
    assert result["median_of_case_minima_gain_db"] == pytest.approx(0.5)
    assert not result["passed"]
    result = w4_audio.decision_gate(
        _matrix([0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, -1.1]), w4_audio.CASES, cfg
    )
    assert not result["passed"]  # median passes, one case regresses >1 dB


def test_gate_rejects_missing_or_undefined_targets():
    cfg = {"minimum_median_gain_db": 1.0, "maximum_case_regression_db": 1.0}
    rows = _matrix([0.0] * 4, [2.0] * 4)
    assert w4_audio.decision_gate(rows, w4_audio.CASES, cfg)["passed"]
    assert (
        w4_audio.decision_gate(rows[:-1], w4_audio.CASES, cfg)["status"]
        == "incomplete_or_nonfinite"
    )
    rows[0]["direct_reference"]["fullband"]["targets"][2]["metrics"] = None
    assert w4_audio.decision_gate(rows, w4_audio.CASES, cfg)["status"] == "incomplete_or_nonfinite"


def test_select_angles_requires_all_four_triple_reverberant_rows():
    rows = [
        {
            "case": case,
            "mode": "reverberant",
            "source_subset": [0, 1, 2],
            "method": "normmusic",
            "peaks": [{"angle_deg": x} for x in [15, -15, 0]],
        }
        for case in w4_audio.CASES
    ]
    assert w4_audio.select_angles(rows, w4_audio.CASES)[w4_audio.CASES[0]] == [-15.0, 0.0, 15.0]
    with pytest.raises(ValueError, match="Incomplete"):
        w4_audio.select_angles(rows[:-1], w4_audio.CASES)
    with pytest.raises(ValueError, match="duplicate"):
        w4_audio.select_angles(rows + [rows[0]], w4_audio.CASES)


def test_design_weights_only_receives_mixture_and_far_field_angles(monkeypatch):
    mixture = np.arange(80.0, dtype=float).reshape(20, 4)
    array = SimpleNamespace(sample_rate_hz=10)
    stft = SimpleNamespace(fft_size=8)
    calls = []

    def covariance(prefix, actual_array, actual_stft):
        assert np.array_equal(prefix, mixture[:10])
        assert actual_array is array and actual_stft is stft
        return np.eye(4)[None].repeat(5, axis=0), 3

    def das(actual_array, actual_stft, angles, **kwargs):
        assert actual_array is array and actual_stft is stft
        assert kwargs == {}  # no source_distances_m, even for oracle angles
        calls.append(("das", angles))
        return np.zeros((5, 4, 3), complex)

    def mpdr(actual_covariance, actual_array, actual_stft, angles, **kwargs):
        assert actual_array is array and actual_stft is stft
        assert kwargs == {
            "diagonal_loading": 0.1,
            "band_hz": (100, 5000),
            "max_condition": 1e6,
            "min_white_noise_gain_db": -10,
        }
        calls.append(("mpdr", angles))
        return SimpleNamespace(
            weights=np.zeros((5, 4, 3), complex),
            status=np.full((5, 3), "mpdr"),
            condition_number=np.zeros(5),
            constraint_residual=np.zeros((5, 3)),
            white_noise_gain_db=np.zeros((5, 3)),
        )

    monkeypatch.setattr(w4_audio, "mixture_covariance", covariance)
    monkeypatch.setattr(w4_audio, "das_weights", das)
    monkeypatch.setattr(w4_audio, "mpdr_weights", mpdr)
    branches = {
        "baseline": [-15.0, 0.0, 15.0],
        "onset": [-14.0, 1.0, 16.0],
        "oracle": [-15.0, 0.0, 15.0],
    }
    designs, _, frames = w4_audio.design_weights(
        mixture,
        array,
        stft,
        branches,
        {
            "calibration_seconds": 1.0,
            "diagonal_loading": 0.1,
            "band_hz": [100, 5000],
            "max_condition": 1e6,
            "min_wng_db": -10,
        },
    )
    assert frames == 3 and len(designs) == 6 and len(calls) == 6


@pytest.mark.parametrize(
    "fault",
    ["missing_band", "missing_reference", "nan", "inactive", "zero_activity", "missing_target"],
)
def test_gate_rejects_invalid_auxiliary_results(fault):
    rows = _matrix([0.0] * 4, [2.0] * 4)
    reference = rows[0]["reverberant_reference"]
    band = reference["evaluation_band"]
    target = band["targets"][0]
    if fault == "missing_band":
        del reference["evaluation_band"]
    elif fault == "missing_reference":
        del rows[0]["reverberant_reference"]
    elif fault == "nan":
        target["metrics"]["output_si_sdr_db"] = float("nan")
    elif fault == "inactive":
        target["status"] = "insufficient_activity"
    elif fault == "zero_activity":
        target["active_samples"] = 0
    else:
        band["targets"].pop()
    result = w4_audio.decision_gate(
        rows, w4_audio.CASES, {"minimum_median_gain_db": 1.0, "maximum_case_regression_db": 1.0}
    )
    assert result["status"] == "incomplete_or_nonfinite"
    assert not result["passed"]


@pytest.mark.parametrize("value", [None, [], "invalid"])
def test_gate_rejects_malformed_reference_or_band(value):
    cfg = {"minimum_median_gain_db": 1.0, "maximum_case_regression_db": 1.0}
    for reference_level in (True, False):
        rows = _matrix([0.0] * 4, [2.0] * 4)
        if reference_level:
            rows[0]["reverberant_reference"] = value
        else:
            rows[0]["reverberant_reference"]["evaluation_band"] = value
        assert (
            w4_audio.decision_gate(rows, w4_audio.CASES, cfg)["status"] == "incomplete_or_nonfinite"
        )
