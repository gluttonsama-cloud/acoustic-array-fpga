"""Independent direct-path and energy-decay checks for the room adapter."""

import numpy as np
import pytest

from acoustic_array.core.config import ArrayConfig
from acoustic_array.core.geometry import steering_vectors
from acoustic_array.evaluation.decay import decay_summary
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomConfig, generate_room_rirs, propagate_room


def test_exponential_decay_time():
    fs, rt = 8000, 0.3
    t = np.arange(fs) / fs
    h = np.exp(-3 * np.log(10) * t / rt)
    r = decay_summary(h, fs)
    for fit in [r["t20"], r["t30"]]:
        assert fit["status"] == "ok"
        assert fit["rt60_s"] == pytest.approx(rt, abs=1e-6)
        assert fit["r_squared"] > 0.999
    assert decay_summary(np.zeros(100), fs)["status"] == "zero_energy"
    assert decay_summary(np.zeros(100), fs)["t20"]["status"] == "zero_energy"
    assert decay_summary([1] + [0] * 99, fs)["t20"]["status"] == "insufficient_decay"


def test_direct_rir_response_and_reference_normalization():
    pytest.importorskip("rir_generator")
    array = ArrayConfig(16000, 4, 0.03, 1)
    room = RoomConfig((4, 4, 3), (2, 1, 1.5), 0.3)
    r = generate_room_rirs(array, [30], [2], room, 0)
    np.testing.assert_array_equal(r.impulse_responses, r.direct_responses)
    n = np.arange(r.direct_responses.shape[1])
    response = np.exp(-2j * np.pi * 1000 * n / 16000) @ r.direct_responses[0]
    expected = steering_vectors(array, [1000], [30], source_distances_m=[2])[0, :, 0]
    np.testing.assert_allclose(response / response[1], expected, atol=2e-4)
    assert abs(response[1]) == pytest.approx(1, abs=2e-4)
    expected_peak = r.direct_distances_m[0] / 343 * 16000
    np.testing.assert_allclose(
        np.argmax(abs(r.direct_responses[0]), axis=0), expected_peak, atol=0.51
    )


def test_room_components_and_direct_mask_scaling():
    pytest.importorskip("rir_generator")
    array = ArrayConfig(8000, 3, 0.03, 1)
    r = generate_room_rirs(array, [0], [1], RoomConfig((3, 3, 2), (1.5, 1, 1), 0.25), 0.12)
    h = r.impulse_responses[0, :, 1]
    assert np.linalg.norm(h - r.direct_responses[0, :, 1]) > 0
    fit = decay_summary(h, 8000)
    assert fit["t20"]["status"] == "ok"
    x = np.random.default_rng(4).normal(size=(1, 8000))
    base, direct = propagate_room(x, array, r)
    calibrated = normalize_active_scene(base, array, activity_components=direct, trim_samples=1000)
    assert calibrated.active_rms[0] == pytest.approx(0.04)
    gain = calibrated.source_gains[0]
    np.testing.assert_allclose(calibrated.scene.components[0], base.components[0] * gain)
    # Linearity and stored RIR preserve independent source contribution.
    delta = np.zeros((1, 8000))
    delta[0, 0] = 1
    propagated, _ = propagate_room(delta, array, r)
    np.testing.assert_allclose(
        propagated.components[0, : len(h)], r.impulse_responses[0], atol=1e-14
    )


@pytest.mark.parametrize("distance", [[True], [np.complex64(1 + 2j)], [0], [-1], [np.nan]])
def test_bad_room_distance(distance):
    with pytest.raises(ValueError):
        generate_room_rirs(ArrayConfig(8000, 3, 0.03, 1), [0], distance, RoomConfig(), 0.2)


def test_room_geometry_and_duration_rejected():
    array = ArrayConfig(8000, 3, 0.03, 1)
    with pytest.raises(ValueError):
        generate_room_rirs(array, [0], [10], RoomConfig(), 0.2)
    with pytest.raises(ValueError):
        generate_room_rirs(array, [0], [1], RoomConfig(rir_duration_s=0.1), 0.2)
    with pytest.raises(ValueError):
        RoomConfig((3, 3, 2), (3, 1, 1))


@pytest.mark.parametrize("interrupt", [False, True])
def test_room_experiment_archive(tmp_path, monkeypatch, interrupt):
    from pathlib import Path

    from acoustic_array.io.artifacts import load_json, sha256_file, write_json
    from acoustic_array.pipelines import room_experiment as experiment

    pytest.importorskip("rir_generator")
    root = Path(__file__).resolve().parents[1]
    cfg = load_json(root / "configs/experiments/g1_room.json")
    cfg.update(
        array_config=str(root / "configs/array/ula16_v1.json"),
        speech_manifest=str(root / "data/manifests/libri_examples.json"),
        duration_s=0.3,
        distances_m=[2],
        adjacent_angles_deg=[30],
        rt60_targets_s=[0],
        fft_sizes=[512],
        trim_samples=2400,
    )
    cfg["room"]["rir_duration_s"] = 0.05
    if interrupt:
        cfg["distances_m"] = [2, 3]
        original = experiment.generate_room_rirs
        calls = 0

        def fail_second(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected interruption")
            return original(*args)

        monkeypatch.setattr(experiment, "generate_room_rirs", fail_second)
    x = np.random.default_rng(2).normal(size=(3, 14400))
    monkeypatch.setattr(experiment, "load_speech_sources", lambda *args: (x, []))
    path = tmp_path / "config.json"
    write_json(path, cfg)
    if interrupt:
        with pytest.raises(RuntimeError, match="injected interruption"):
            experiment.run_room_experiment(path, tmp_path)
        run = next(tmp_path.glob("g1_room_*"))
        report = load_json(run / "metrics.json")
        assert report["run_status"] == "in_progress"
        assert report["expected_cases"] == 2
        assert report["completed_cases"] == 1
        assert not (run / "manifest.json").exists()
        return
    run = experiment.run_room_experiment(path, tmp_path)
    report = load_json(run / "metrics.json")
    assert report["run_status"] == "complete"
    assert report["expected_cases"] == report["completed_cases"] == 1
    rows = report["rows"]
    assert len(rows) == 2
    for row in rows:
        assert row["direct_reference"] == row["reverberant_reference"]
        for band in ["fullband", "evaluation_band"]:
            assert row["direct_reference"][band]["failed_targets"] == 0
    manifest = load_json(run / "manifest.json")
    for name, digest in manifest["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
    for name, digest in manifest["source_code_files_sha256"].items():
        assert sha256_file(run / "source_snapshot" / name) == digest
