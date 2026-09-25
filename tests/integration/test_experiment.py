import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from acoustic_array.io.artifacts import load_json, sha256_file, write_pcm16
from acoustic_array.pipelines.experiment import run_experiment


def test_experiment_artifacts_and_hashes(project_root, tmp_path):
    config = load_json(project_root / "configs/experiments/m1_das.json")
    config["array_config"] = str(project_root / "configs/array/ula16_v1.json")
    config["duration_s"] = 0.15
    config["evaluation_trim_s"] = 0.02
    config["fft_sizes"] = [256]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run = run_experiment(config_path, tmp_path / "runs")
    manifest = load_json(run / "manifest.json")
    for name, digest in manifest["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
    rate, audio = wavfile.read(run / "mixture_16ch.wav")
    assert rate == 48000 and audio.shape == (7200, 16)
    assert load_json(run / "metrics.json")["status"] == "engineering_only_not_B1_or_B2"
    repeat = run_experiment(config_path, tmp_path / "runs")
    assert repeat != run
    np.testing.assert_array_equal(np.load(run / "das_256.npy"), np.load(repeat / "das_256.npy"))
    assert load_json(repeat / "manifest.json")["config_sha256"] == manifest["config_sha256"]


def test_export_does_not_silently_normalize(tmp_path):
    with pytest.raises(ValueError, match="clip"):
        write_pcm16(Path(tmp_path) / "bad.wav", 48000, [0, 2])


def test_perfect_single_source_is_reported_without_fabricated_score(project_root, tmp_path):
    config = load_json(project_root / "configs/experiments/m1_das.json")
    config.update(
        {
            "array_config": str(project_root / "configs/array/ula16_v1.json"),
            "source_angles_deg": [0],
            "noise_rms": 0,
            "duration_s": 0.15,
            "evaluation_trim_s": 0.02,
            "fft_sizes": [256],
            "evaluation_band_hz": [800, 4000],
        }
    )
    config_path = tmp_path / "single.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    run = run_experiment(config_path, tmp_path / "runs")
    metrics = load_json(run / "metrics.json")
    score = metrics["rows"][0]["fullband"]
    assert score["input_si_sdr_db"] is None
    assert score["metric_status"]["input_si_sdr_db"] == "positive_infinity"
    assert metrics["evaluation_band_hz"] == [800, 4000]
    assert "evaluation_band" in metrics["rows"][0]
    assert (run / "manifest.json").exists()
