import json

import numpy as np
import pytest
import soundfile as sf

from acoustic_array.io.artifacts import load_json, sha256_file
from acoustic_array.io.speech import load_speech_sources
from acoustic_array.pipelines.active_scene_experiment import run_active_scene_experiment
from acoustic_array.pipelines.activity_experiment import run_activity_experiment
from acoustic_array.pipelines.component_experiment import (
    _target_diagnostics,
    run_component_experiment,
)
from acoustic_array.pipelines.mismatch_experiment import run_mismatch_experiment
from acoustic_array.pipelines.speech_experiment import run_speech_experiment


@pytest.fixture
def tiny_sources(tmp_path):
    # Test-generated tones, not third-party speech. No network needed for tests.
    records = []
    for k in range(3):
        path = tmp_path / f"{k}.wav"
        time = np.arange(4000) / 16000
        pcm = 0.1 * np.sin(2 * np.pi * (900 + k * 400) * time)
        pcm *= 0.6 + 0.4 * np.sin(2 * np.pi * 17 * time) ** 2
        sf.write(path, pcm, 16000, subtype="FLOAT")
        records.append(
            {
                "path": path.name,
                "speaker_id": str(k),
                "sha256": sha256_file(path),
                "attribution": "test generated tone",
                "url": "test:generated",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"license": "CC-BY-4.0", "files": records}), encoding="utf-8")
    return manifest


def test_speech_loading_resampling_and_tamper_detection(tiny_sources):
    data, records = load_speech_sources(tiny_sources, 48000, 0.2, 0.04)
    assert data.shape == (3, 9600)
    np.testing.assert_allclose(np.sqrt(np.mean(data**2, axis=1)), 0.04, atol=1e-12)
    assert records[0]["resample_up"] == 3 and records[0]["resample_down"] == 1
    with pytest.raises(ValueError, match="short"):
        load_speech_sources(tiny_sources, 48000, 1)
    raw = load_json(tiny_sources)
    raw["files"][0]["sha256"] = "0" * 64
    tiny_sources.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_speech_sources(tiny_sources, 48000, 0.2)


def test_m2_artifact_contract(project_root, tmp_path, tiny_sources):
    config = load_json(project_root / "configs/experiments/m2_speech_development.json")
    config.update(
        {
            "array_config": str(project_root / "configs/array/ula16_v1.json"),
            "speech_manifest": str(tiny_sources),
            "duration_s": 0.2,
            "evaluation_trim_s": 0.02,
            "fft_sizes": [256],
            "direction_offsets_deg": [0, 1, 1.0000001],
        }
    )
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    run = run_speech_experiment(path, tmp_path / "runs")
    metrics = load_json(run / "metrics.json")
    assert len(metrics["rows"]) == 18
    assert metrics["status"] == "development_only_not_B1_or_B2"
    assert metrics["constraint_diagnostics"][0]["fallback_bin_outputs"] == 0
    for name, digest in load_json(run / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
    _, rate = sf.read(run / "fixed_constraint_256_target_0.wav")
    assert rate == 48000
    assert "test generated tone" in (run / "REPORT.md").read_text(encoding="utf-8")
    weight_files = [r["weights_file"] for r in metrics["constraint_diagnostics"]]
    assert len(set(weight_files)) == 3
    for index, name in enumerate(weight_files):
        assert np.load(run / name)["direction_offset_deg"] == [0, 1, 1.0000001][index]
    audit = run_component_experiment(run, tmp_path / "audit")
    activity_run = run_activity_experiment(run, tmp_path / "activity")
    active = load_json(activity_run / "metrics.json")
    assert len(active["rows"]) == 6
    assert active["scene_statistics"]["das"]["effect_scenes"] == 1
    for target in range(3):
        pair = [row for row in active["rows"] if row["target"] == target]
        assert pair[0]["active_fraction_after_trim"] == pair[1]["active_fraction_after_trim"]
    masks = np.load(activity_run / "reference_masks.npy")
    assert masks.shape == (9600, 3) and masks.dtype == np.bool_
    for row in active["rows"]:
        old = next(
            r
            for r in metrics["rows"]
            if r["direction_offset_deg"] == 0
            and r["target"] == row["target"]
            and r["method"] == row["method"]
        )
        assert row["fullband"]["whole"] == old["fullband"]
    audit_metrics = load_json(audit / "metrics.json")
    assert audit_metrics["reconstruction_relative_l2"] < 1e-12
    assert len(audit_metrics["rows"]) == 3
    with np.load(audit / "components.npz") as outputs:
        rebuilt = outputs["source_outputs"].sum(axis=0) + outputs["noise_output"]
    np.testing.assert_allclose(rebuilt, np.load(run / "fixed_constraint_256.npy"), atol=1e-14)
    for name, digest in load_json(audit / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(audit / name) == digest
    with (run / "input.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_component_experiment(run, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_no_bins_fails_before_output_creation(project_root, tmp_path, tiny_sources):
    config = load_json(project_root / "configs/experiments/m2_speech_development.json")
    config.update(
        {
            "array_config": str(project_root / "configs/array/ula16_v1.json"),
            "speech_manifest": str(tiny_sources),
            "constraint_band_hz": [501, 502],
            "fft_sizes": [256],
        }
    )
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="no FFT bins"):
        run_speech_experiment(path, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_silent_component_diagnostics():
    silent = np.zeros(2000)
    noise = np.random.default_rng(7).normal(size=2000)
    diagnostics = _target_diagnostics(
        [silent, noise, silent], [silent, noise, silent], 48000, 100, (500, 5000)
    )
    for group in diagnostics["metrics"].values():
        assert group["target_only_si_sdr"]["status"] == "undefined_silent_reference"
        assert group["output"]["sir"]["status"] == "negative_infinity"
        assert group["output"]["snr"]["status"] == "undefined"


def test_active_scene_artifacts(project_root, tmp_path, tiny_sources):
    cfg = load_json(project_root / "configs/experiments/m2_active_scene_development.json")
    cfg.update(
        array_config=str(project_root / "configs/array/ula16_v1.json"),
        speech_manifest=str(tiny_sources),
        duration_s=0.2,
        trim_samples=960,
        fft_sizes=[256],
    )
    path = tmp_path / "active.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    run = run_active_scene_experiment(path, tmp_path / "active_runs")
    m = load_json(run / "metrics.json")
    np.testing.assert_allclose(m["source_active_rms"], 0.04, rtol=1e-12)
    assert m["reference_noise_rms"] == pytest.approx(0.004)
    assert len(m["rows"]) == 6
    from acoustic_array.core.config import ArrayConfig
    from acoustic_array.simulation.active_scene import build_active_scene

    with np.load(run / "reconstruction_input.npz") as saved:
        rebuilt = build_active_scene(
            saved["sources"],
            ArrayConfig.from_mapping(load_json(project_root / "configs/array/ula16_v1.json")),
            cfg["angles_deg"],
            active_rms=cfg["active_rms"],
            noise_snr_db=cfg["noise_snr_db"],
            trim_samples=cfg["trim_samples"],
            seed=cfg["seed"],
        )
        np.testing.assert_array_equal(rebuilt.masks, saved["masks"])
        np.testing.assert_array_equal(rebuilt.source_gains, saved["source_gains"])
        np.testing.assert_allclose(rebuilt.active_rms, 0.04, rtol=1e-12)
    for name, digest in load_json(run / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(run / name) == digest


def test_activity_silence_is_explicitly_ineligible(project_root, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    config = load_json(project_root / "configs/experiments/m2_speech_development.json")
    config.update(fft_sizes=[256], evaluation_trim_s=0.02)
    expanded = {
        "experiment": config,
        "array": load_json(project_root / "configs/array/ula16_v1.json"),
    }
    (parent / "config.expanded.json").write_text(json.dumps(expanded), encoding="utf-8")
    np.savez(parent / "input.npz", components=np.zeros((3, 9600, 16)), mixture=np.zeros((9600, 16)))
    for method in ["das", "fixed_constraint"]:
        np.save(parent / f"{method}_256.npy", np.zeros((9600, 3)))
    manifest = {"outputs_sha256": {p.name: sha256_file(p) for p in parent.iterdir()}}
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    run = run_activity_experiment(parent, tmp_path / "activity")
    metrics = load_json(run / "metrics.json")
    assert metrics["scene_statistics"]["das"]["status"] == "ineligible_reference_activity"
    assert all(r["fullband"]["whole"] is None for r in metrics["rows"])
    assert all(
        r["fullband"]["active"]["status"] == "insufficient_activity" for r in metrics["rows"]
    )


def test_mismatch_replay_and_config_validation(project_root, tmp_path, tiny_sources):
    config = load_json(project_root / "configs/experiments/m2_speech_development.json")
    config.update(
        array_config=str(project_root / "configs/array/ula16_v1.json"),
        speech_manifest=str(tiny_sources),
        duration_s=0.2,
        evaluation_trim_s=0.02,
        fft_sizes=[256],
        direction_offsets_deg=[0],
    )
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    parent = run_speech_experiment(path, tmp_path / "parents")
    case = {
        "name": "test",
        "direction_errors_deg": [-1, 0, 1],
        "gain_db": [0] * 16,
        "delay_samples": [0] * 16,
    }
    case["gain_db"][0] = 1
    case["delay_samples"][2] = 0.5
    sweep = {
        "schema_version": "0.1",
        "parent_run": str(parent),
        "fft_sizes": [256],
        "cases": [case],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(sweep), encoding="utf-8")
    run = run_mismatch_experiment(path, tmp_path / "sweeps")
    metrics = load_json(run / "metrics.json")
    assert len(metrics["rows"]) == 6
    assert metrics["diagnostics"][0]["reconstruction_relative_l2"] < 1e-12
    for name, digest in load_json(run / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
    case["gain_db"][7] = 1
    path.write_text(json.dumps(sweep), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration anchor"):
        run_mismatch_experiment(path, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    case["gain_db"][7] = 0
    path.write_text(json.dumps(sweep), encoding="utf-8")
    expanded = load_json(parent / "config.expanded.json")
    expanded["experiment"]["source_angles_deg"].append(70)
    (parent / "config.expanded.json").write_text(json.dumps(expanded), encoding="utf-8")
    manifest = load_json(parent / "manifest.json")
    manifest["outputs_sha256"]["config.expanded.json"] = sha256_file(
        parent / "config.expanded.json"
    )
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="component shapes"):
        run_mismatch_experiment(path, tmp_path / "invalid_shapes")
    assert not (tmp_path / "invalid_shapes").exists()
    expanded["experiment"]["source_angles_deg"].pop()
    (parent / "config.expanded.json").write_text(json.dumps(expanded), encoding="utf-8")
    with np.load(parent / "input.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    data["components"][0] = 0
    data["mixture"] = data["components"].sum(axis=0) + data["noise"]
    np.savez_compressed(parent / "input.npz", **data)
    for name in ["config.expanded.json", "input.npz"]:
        manifest["outputs_sha256"][name] = sha256_file(parent / name)
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="non-silent"):
        run_mismatch_experiment(path, tmp_path / "silent")
    assert not (tmp_path / "silent").exists()
