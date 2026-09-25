"""Pre-score regression: fixed-composition statistics and batch failure retention."""

from pathlib import Path

import pytest

from acoustic_array.evaluation.scene_statistics import summarize_s1_stratified
from acoustic_array.io.artifacts import load_json, write_json
from acoustic_array.pipelines import holdout

PROJECT = Path(__file__).resolve().parents[1]


def test_stratified_bootstrap_keeps_composition():
    rows = [
        {
            "scene_id": str(i),
            "status": "ok",
            "stratum": "a" if i < 10 else "b",
            "target_si_sdri_db": [0 if i < 10 else 100] * 3,
        }
        for i in range(20)
    ]
    result = summarize_s1_stratified(rows, {"a": 10, "b": 10}, bootstrap_samples=200)
    assert [r["db"] for r in result["median_ci95"]] == [0, 0]
    assert result["median"]["db"] == 0
    rows[0] = {**rows[0], "status": "failed"}
    result = summarize_s1_stratified(rows, {"a": 10, "b": 10}, bootstrap_samples=200)
    assert result["effect_scenes"] == 20 and result["failed_scenes"] == 1
    assert result["scene_scores"][0]["status"] == "negative_infinity"
    with pytest.raises(ValueError, match="counts"):
        summarize_s1_stratified(rows[:-1], {"a": 10, "b": 10})
    rows[0]["status"] = "silent"
    with pytest.raises(ValueError, match="silent"):
        summarize_s1_stratified(rows, {"a": 10, "b": 10})


def test_all_failed_strata_have_explicit_infinite_intervals():
    rows = [{"scene_id": str(i), "stratum": "s", "status": "failed"} for i in range(3)]
    r = summarize_s1_stratified(rows, {"s": 3}, bootstrap_samples=20)
    assert r["failure_fraction"] == 1
    assert all(v["status"] == "negative_infinity" for v in r["p10_ci95"])


def test_batch_preserves_failed_scene_and_does_not_select_secondary(tmp_path, monkeypatch):
    for name in [
        "configs/experiments/s1_holdout_preregistered_v1.json",
        "configs/array/ula16_v1.json",
        "data/manifests/s1_disjoint_v1.json",
        "data/manifests/split_registry.json",
    ]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT / name).read_bytes())
    seeds = []

    def fake_kernel(config, output):
        cfg = load_json(config)
        seeds.append(cfg["seed"])
        if len(seeds) == 1:
            raise ValueError("corrupt/silent source fixture")
        child = output / "fake_kernel"
        child.mkdir()
        rows = []
        for n in [512, 1024]:
            for method in ["fixed_constraint", "das"]:
                for target in range(3):
                    value = 5 if n == 512 else 30
                    rows.append(
                        {
                            "fft_size": n,
                            "method": method,
                            "target": target,
                            "evaluation_band": {
                                "active": {
                                    "status": "ok",
                                    "metrics": {
                                        "si_sdri_db": value,
                                        "metric_status": {"si_sdri_db": "finite"},
                                    },
                                }
                            },
                        }
                    )
        write_json(child / "metrics.json", {"rows": rows})
        return child

    monkeypatch.setattr(holdout, "run_active_scene_experiment", fake_kernel)
    config = tmp_path / "configs/experiments/s1_holdout_preregistered_v1.json"
    run = holdout.run_holdout(config, tmp_path / "runs", tmp_path)
    result = load_json(run / "results.json")
    assert seeds == list(range(20260922, 20260952))
    assert result["b1_effect_thresholds_met"] is False
    assert result["summary"]["secondary"]["median"]["db"] == 30
    for rows in result["records"].values():
        assert len(rows) == 30 and rows[0]["status"] == "failed"
    assert result["summary"]["primary"]["failed_scenes"] == 1
    # Frozen manifest identity is checked before a run is created.
    corpus_path = tmp_path / "data/manifests/s1_disjoint_v1.json"
    corpus_path.write_text("{}")
    with pytest.raises(ValueError, match="corpus hash"):
        holdout.run_holdout(config, tmp_path / "runs", tmp_path)
    changed = load_json(config)
    changed["statistics"]["bootstrap_scheme"] = "ordinary"
    write_json(config, changed)
    with pytest.raises(ValueError, match="preregistration hash"):
        holdout.run_holdout(config, tmp_path / "runs", tmp_path)


def test_design_and_output_contract_rejects_drift():
    cfg = load_json(PROJECT / "configs/experiments/s1_holdout_preregistered_v1.json")
    corpus = load_json(PROJECT / "data/manifests/s1_disjoint_v1.json")
    holdout.validate_design(cfg, corpus)
    cfg["primary"]["fft_size"] = 1024
    with pytest.raises(ValueError, match="method"):
        holdout.validate_design(cfg, corpus)
    with pytest.raises(ValueError, match="Missing"):
        holdout.score_rows([], 512, "das")
