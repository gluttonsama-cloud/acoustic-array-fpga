import numpy as np
import pytest

from acoustic_array.beamforming.das import das_weights
from acoustic_array.beamforming.mpdr import mixture_covariance, mpdr_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import steering_vectors

ARRAY = ArrayConfig(16000, 4, 0.03, 1)
STFT = STFTConfig(128, 64)


def test_identity_and_zero_covariance():
    r = np.broadcast_to(np.eye(4), (65, 4, 4)).copy()
    das = das_weights(ARRAY, STFT, [-30, 30])
    result = mpdr_weights(r, ARRAY, STFT, [-30, 30])
    np.testing.assert_allclose(result.weights, das, atol=1e-15)
    zero = mpdr_weights(r * 0, ARRAY, STFT, [-30, 30])
    np.testing.assert_array_equal(zero.weights, das)
    assert np.any(zero.status == "fallback_zero_power")


def test_analytic_diagonal_inverse_and_scaling():
    r = np.broadcast_to(np.diag([1, 2, 4, 8]), (65, 4, 4)).copy()
    result = mpdr_weights(r, ARRAY, STFT, [20], diagonal_loading=0.1)
    a = steering_vectors(ARRAY, [1000], [20])[0, :, 0]
    u = a / (np.array([1, 2, 4, 8]) + 0.1 * 3.75)
    expected = u / np.vdot(a, u)
    np.testing.assert_allclose(result.weights[8, :, 0], expected, atol=1e-14)
    scaled = mpdr_weights(r * 1e-8, ARRAY, STFT, [20], diagonal_loading=0.1)
    np.testing.assert_allclose(result.weights, scaled.weights, atol=1e-14)
    assert np.nanmax(result.constraint_residual) < 1e-12
    fallback = mpdr_weights(r, ARRAY, STFT, [20], min_white_noise_gain_db=100)
    np.testing.assert_array_equal(fallback.weights, das_weights(ARRAY, STFT, [20]))


def test_covariance_conjugation_and_block_independence():
    n = np.arange(4096)
    phases = np.array([0, 0.3, -0.2, 0.9])
    x = np.cos(2 * np.pi * 1000 * n[:, None] / 16000 + phases)
    r, count = mixture_covariance(x, ARRAY, STFT, block_size=177)
    other, n2 = mixture_covariance(x, ARRAY, STFT, block_size=301)
    np.testing.assert_array_equal(r, other)
    assert count == n2 == 63
    expected = np.exp(1j * (phases[:, None] - phases[None, :]))
    np.testing.assert_allclose(r[8] / r[8, 0, 0], expected, atol=1e-12)
    with pytest.raises(ValueError):
        mixture_covariance(x[:128], ARRAY, STFT)


def test_interferer_suppression():
    a = steering_vectors(ARRAY, np.fft.rfftfreq(128, 1 / 16000), [40])[:, :, 0]
    r = np.eye(4)[None] + 100 * a[:, :, None] * a[:, None, :].conj()
    result = mpdr_weights(r, ARRAY, STFT, [-40], diagonal_loading=0.01)
    das = das_weights(ARRAY, STFT, [-40])
    assert abs(np.vdot(result.weights[16, :, 0], a[16])) < 0.1 * abs(np.vdot(das[16, :, 0], a[16]))


@pytest.mark.parametrize("kind", ["nonhermitian", "negative", "nan"])
def test_invalid_covariance(kind):
    r = np.broadcast_to(np.eye(4), (65, 4, 4)).astype(complex).copy()
    if kind == "nonhermitian":
        r[1, 0, 1] = 1j
    if kind == "negative":
        r[1, 0, 0] = -1
    if kind == "nan":
        r[1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        mpdr_weights(r, ARRAY, STFT, [0])


def test_calibration_runner_rebuild_and_archive(tmp_path, monkeypatch):
    from pathlib import Path

    from acoustic_array.io.artifacts import load_json, sha256_file, write_json
    from acoustic_array.pipelines import mpdr_experiment, room_experiment

    pytest.importorskip("rir_generator")
    root = Path(__file__).resolve().parents[1]
    cfg = load_json(root / "configs/experiments/g1_room.json")
    cfg.update(
        array_config=str(root / "configs/array/ula16_v1.json"),
        speech_manifest=str(root / "data/manifests/libri_examples.json"),
        duration_s=0.5,
        distances_m=[2],
        adjacent_angles_deg=[30],
        rt60_targets_s=[0],
        fft_sizes=[512],
        trim_samples=2400,
    )
    cfg["room"]["rir_duration_s"] = 0.05
    x = np.random.default_rng(3).normal(size=(3, 24000))
    monkeypatch.setattr(room_experiment, "load_speech_sources", lambda *args: (x, []))
    p = tmp_path / "room.json"
    write_json(p, cfg)
    parent = room_experiment.run_room_experiment(p, tmp_path)
    cfg2 = load_json(root / "configs/experiments/g1_mpdr.json")
    cfg2.update(room_run=str(parent), calibration_seconds=0.15)
    p2 = tmp_path / "mpdr.json"
    write_json(p2, cfg2)
    original = mpdr_experiment.mixture_covariance
    seen = []

    def inspect_prefix(pcm, array, stft):
        seen.append(pcm.shape)
        return original(pcm, array, stft)

    monkeypatch.setattr(mpdr_experiment, "mixture_covariance", inspect_prefix)
    run = mpdr_experiment.run_mpdr_experiment(p2, tmp_path)
    assert seen == [(7200, 16)]
    e = load_json(run / "config.expanded.json")
    assert e["calibration_interval_samples"] == [0, 7200]
    assert e["evaluation_interval_samples"] == [9600, 21600]
    report = load_json(run / "metrics.json")
    assert report["run_status"] == "results_complete" and report["completed_cases"] == 1
    complete = load_json(run / "completion.json")
    assert complete["run_status"] == "complete"
    assert complete["manifest_sha256"] == sha256_file(run / "manifest.json")
    assert len(report["rows"]) == 4
    for row in report["rows"]:
        assert row["direct_reference"]["fullband"]["failed_targets"] == 0
        for t in row["direct_reference"]["fullband"]["targets"]:
            assert t["active_samples"] <= 12000
    for name, digest in load_json(run / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
    original_write = mpdr_experiment.write_json

    def fail_manifest(path, value):
        if path.name == "manifest.json":
            raise OSError("injected archive failure")
        return original_write(path, value)

    monkeypatch.setattr(mpdr_experiment, "write_json", fail_manifest)
    failed_root = tmp_path / "failed"
    with pytest.raises(OSError, match="injected archive failure"):
        mpdr_experiment.run_mpdr_experiment(p2, failed_root)
    failed_run = next(failed_root.glob("g1_mpdr_*"))
    assert not (failed_run / "completion.json").exists()
    assert load_json(failed_run / "metrics.json")["run_status"] == "results_complete"
    with (parent / "sources.npz").open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        mpdr_experiment.run_mpdr_experiment(p2, tmp_path)
