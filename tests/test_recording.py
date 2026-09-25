import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from acoustic_array.io.artifacts import sha256_file
from acoustic_array.pipelines.recording import replay_recording


@pytest.fixture
def recording(tmp_path):
    root = Path(__file__).resolve().parents[1]
    array = json.loads((root / "configs/array/w3_staggered.json").read_text())
    (tmp_path / "array.json").write_text(json.dumps(array))
    profile = {
        "array_config": "array.json",
        "wav_channel_for_mic": list(range(15, -1, -1)),
        "geometry_evidence": {"kind": "synthetic", "note": "IO regression, not a recording"},
    }
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    rng = np.random.default_rng(73)
    pcm = rng.normal(0, 0.005, (18000, 16))
    sf.write(tmp_path / "input.wav", pcm, 48000, subtype="FLOAT")
    return tmp_path, pcm, profile


def run(recording, **extra):
    path = recording[0]
    kwargs = {"calibration_start": 0.05, "calibration_seconds": 0.15, "angles": [0.0]}
    kwargs.update(extra)
    return replay_recording(path / "input.wav", path / "profile.json", path / "runs", **kwargs)


def test_manual_file_replay_mapping_prefix_and_frozen_evidence(recording):
    path, _, _ = recording
    original, _ = sf.read(path / "input.wav")
    result = run(recording)
    report = json.loads((result / "report.json").read_text())
    reference, sr = sf.read(result / "reference_raw.wav")
    # MIC7 maps to input WAV channel8; float32 source and export preserve samples.
    np.testing.assert_array_equal(reference, original[:, 8])
    assert sr == 48000 and report["calibration_samples"] == [2400, 9600]
    assert report["status"] == "manual_direction_replay"
    assert report["candidate_identity_verified"] is False
    with np.load(result / "outputs.npz") as z:
        assert z["output"].shape == (18000, 1)
        assert not z["output"][:9600].any()
    manifest = json.loads((result / "manifest.json").read_text())
    assert all(sha256_file(result / n) == h for n, h in manifest["outputs_sha256"].items())


@pytest.mark.parametrize("fault", ["zero", "duplicate", "clipped"])
def test_anomaly_keeps_reference_without_enhanced_outputs(recording, fault):
    path, pcm, _ = recording
    if fault == "zero":
        pcm[:, 0] = 0
    elif fault == "duplicate":
        pcm[:, 0] = pcm[:, 1]
    else:
        pcm[0, 0] = 1
    sf.write(path / "input.wav", pcm, 48000, subtype="FLOAT")
    result = run(recording)
    assert (
        json.loads((result / "report.json").read_text())["status"]
        == "reference_only_channel_anomaly"
    )
    assert (result / "reference_raw.wav").exists()
    assert not (result / "outputs.npz").exists()


def test_silent_calibration_does_not_invent_candidates(recording):
    path, pcm, _ = recording
    pcm[2400:9600] = 0
    sf.write(path / "input.wav", pcm, 48000, subtype="FLOAT")
    result = run(recording, angles=None, source_count=3)
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "reference_only_silent_calibration"
    assert report["candidate_angles_deg"] == []


def test_automatic_empty_candidate_output_is_reference_only(recording, monkeypatch):
    from types import SimpleNamespace

    import acoustic_array.localization.music_reference as music

    monkeypatch.setattr(
        music, "music_candidates", lambda *args, **kwargs: SimpleNamespace(peaks=[])
    )
    result = run(recording, angles=None, source_count=2)
    report = json.loads((result / "report.json").read_text())
    assert report["status"] == "reference_only_no_candidates"
    assert report["source_count_prior"] == 2


@pytest.mark.parametrize(
    "fault", ["rate", "channels", "mapping", "implicit_geometry", "interval", "mode"]
)
def test_rejects_invalid_contract_before_run_creation(recording, fault):
    path, pcm, profile = recording
    extra = {}
    if fault == "rate":
        sf.write(path / "input.wav", pcm, 44100, subtype="FLOAT")
    elif fault == "channels":
        sf.write(path / "input.wav", pcm[:, :2], 48000, subtype="FLOAT")
    elif fault == "mapping":
        profile["wav_channel_for_mic"] = [0] * 16
        (path / "profile.json").write_text(json.dumps(profile))
    elif fault == "implicit_geometry":
        a = json.loads((path / "array.json").read_text())
        a["geometry"]["kind"] = "ula"
        (path / "array.json").write_text(json.dumps(a))
    elif fault == "interval":
        extra["calibration_seconds"] = 1.0
    else:
        extra["source_count"] = 3
    with pytest.raises(ValueError):
        run(recording, **extra)
    assert not (path / "runs").exists()


def test_input_change_during_read_rejected_before_output(recording, monkeypatch):
    path, _, _ = recording
    read = sf.read

    def mutating_read(file, **kwargs):
        result = read(file, **kwargs)
        with open(file, "ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(sf, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        run(recording)
    assert not (path / "runs").exists()


def test_silent_auto_reference_does_not_require_optional_dependency(recording, monkeypatch):
    import importlib.metadata

    path, pcm, _ = recording
    pcm[2400:9600] = 0
    sf.write(path / "input.wav", pcm, 48000, subtype="FLOAT")
    version = importlib.metadata.version

    def missing_optional(package):
        if package == "pyroomacoustics":
            raise importlib.metadata.PackageNotFoundError(package)
        return version(package)

    monkeypatch.setattr(importlib.metadata, "version", missing_optional)
    result = run(recording, angles=None, source_count=1)
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["dependencies"]["pyroomacoustics"] is None
    assert (result / "completion.json").exists()
