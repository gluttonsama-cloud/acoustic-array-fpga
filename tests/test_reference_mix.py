import numpy as np
import pytest

from acoustic_array.beamforming.reference_mix import mix_reference


def test_endpoints_same_target_and_antiphase_counterexample():
    x = np.array([1.0, -2.0, 3.0])
    y = np.column_stack([x, -x])
    np.testing.assert_array_equal(mix_reference(y, x, 0), y)
    np.testing.assert_array_equal(mix_reference(y, x, 1), np.column_stack([x, x]))
    z = mix_reference(y, x, 0.5)
    np.testing.assert_array_equal(z[:, 0], x)
    np.testing.assert_array_equal(z[:, 1], 0)  # deliberately no universal protection claim


def test_block_partition_and_extreme_finite_inputs():
    x = np.arange(25.0)
    y = np.column_stack([x[::-1], x * 2])
    chunks = np.concatenate([mix_reference(y[:7], x[:7], 0.5), mix_reference(y[7:], x[7:], 0.5)])
    np.testing.assert_array_equal(chunks, mix_reference(y, x, 0.5))
    largest = np.finfo(float).max
    assert np.isfinite(mix_reference([[largest]], [largest], 0.5)).all()


@pytest.mark.parametrize("fraction", [-0.1, 1.1, True, float("nan")])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        mix_reference([[1.0]], [1.0], fraction)


def test_rejects_misaligned_nonfinite_and_complex():
    for y, x in [([[1.0]], [1.0, 2.0]), ([[float("nan")]], [1.0]), ([[1j]], [1.0])]:
        with pytest.raises(ValueError):
            mix_reference(y, x, 0.5)


@pytest.mark.parametrize(
    "fault,expected",
    [("none", True), ("one_loss", False), ("no_utility", False), ("auxiliary_missing", False)],
)
def test_protection_gate_checks_every_target_and_requires_utility(fault, expected):
    from tools.w4_reference_mix import CASES, protection_gate

    rows = []
    for case in CASES:
        row = {
            "case": case,
            "assignment_scoring_only": [{"localized_within_tolerance": True} for _ in range(3)],
        }
        for reference in ("direct_reference", "reverberant_reference"):
            row[reference] = {}
            for band in ("fullband", "evaluation_band"):
                value = 0.0 if fault == "no_utility" else 4.0
                row[reference][band] = {
                    "targets": [
                        {
                            "status": "ok",
                            "active_samples": 1000,
                            "metrics": {
                                "input_si_sdr_db": -10.0,
                                "output_si_sdr_db": value - 10.0,
                                "si_sdri_db": value,
                            },
                        }
                        for _ in range(3)
                    ]
                }
        rows.append(row)
    if fault == "one_loss":
        rows[0]["direct_reference"]["evaluation_band"]["targets"][2]["metrics"][
            "si_sdri_db"
        ] = -1.01
    elif fault == "auxiliary_missing":
        del rows[0]["reverberant_reference"]["fullband"]
    result = protection_gate(
        rows,
        {"minimum_target_improvement_db": -1.0, "minimum_localized_median_improvement_db": 3.0},
    )
    assert result["passed"] is expected
