"""Independent spherical-wave checks, including time-domain phase and far limit."""

import numpy as np
import pytest

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import steering_vectors
from acoustic_array.simulation.active_scene import build_active_scene
from acoustic_array.simulation.near_field import simulate_near_field

ARRAY = ArrayConfig(48000, 16, 0.03048, 7)


def test_broadside_symmetry_and_reference():
    a = steering_vectors(ARRAY, [0, 1000], [0], source_distances_m=[2])
    np.testing.assert_allclose(a[:, :, 0], a[:, ::-1, 0], atol=1e-14)
    np.testing.assert_array_equal(a[:, 7, 0], 1)
    assert a[0, 0, 0].real < 1  # outer microphone farther than physical reference


def test_time_domain_tone_matches_manifold():
    n = np.arange(48000)
    x = np.cos(2 * np.pi * 1500 * n / 48000)
    scene = simulate_near_field([x], ARRAY, [35], [2])
    phasors = 2 * np.mean(
        scene.components[0, 4800:-4800] * np.exp(-2j * np.pi * 1500 * n[4800:-4800, None] / 48000),
        axis=0,
    )
    expected = steering_vectors(ARRAY, [1500], [35], source_distances_m=[2])[0, :, 0]
    np.testing.assert_allclose(phasors, expected, atol=3e-5)
    np.testing.assert_allclose(scene.components[0, :, 7], x, atol=1e-14)


def test_far_limit_and_default_regression():
    plane = steering_vectors(ARRAY, [500, 5000], [-35, 0, 40])
    far = steering_vectors(ARRAY, [500, 5000], [-35, 0, 40], source_distances_m=[1e6] * 3)
    np.testing.assert_allclose(far, plane, atol=3e-6)
    stft = STFTConfig()
    expected = steering_vectors(ARRAY, np.fft.rfftfreq(512, 1 / 48000), [-35, 0, 40]) / 16
    expected[[0, -1]] = expected[[0, -1]].real
    np.testing.assert_array_equal(das_weights(ARRAY, stft, [-35, 0, 40]), expected)


def test_matched_weights_and_fallback():
    angles, distances = [-30, 0, 30], [2, 3, 2]
    stft = STFTConfig()
    a = steering_vectors(
        ARRAY, np.fft.rfftfreq(512, 1 / 48000), angles, source_distances_m=distances
    )
    das = das_weights(ARRAY, stft, angles, source_distances_m=distances)
    np.testing.assert_allclose(np.sum(a[1:-1].conj() * das[1:-1], axis=1), 1, atol=1e-14)
    result = constrained_weights(ARRAY, stft, angles, source_distances_m=distances)
    assert np.any(result.status == "constrained")
    assert np.max(result.constraint_residual[result.status == "constrained"]) < 1e-12
    fallback = constrained_weights(
        ARRAY, stft, angles, source_distances_m=distances, min_white_noise_gain_db=100
    )
    np.testing.assert_array_equal(fallback.weights, das)


@pytest.mark.parametrize(
    "distances",
    [
        [0],
        [-1],
        [np.nan],
        [np.inf],
        [2, 3],
        2,
        [True],
        [np.bool_(True)],
        [1 + 0j],
        [np.complex64(1 + 2j)],
    ],
)
def test_invalid_distance(distances):
    with pytest.raises(ValueError):
        steering_vectors(ARRAY, [1000], [0], source_distances_m=distances)
    with pytest.raises(ValueError):
        simulate_near_field([[1, 2, 3]], ARRAY, [0], distances)


def test_source_at_microphone_rejected():
    with pytest.raises(ValueError):
        steering_vectors(ARRAY, [1000], [90], source_distances_m=[7.5 * ARRAY.spacing_m])
    with pytest.raises(ValueError):
        simulate_near_field([[1, 2]], ARRAY, [90], [7.5 * ARRAY.spacing_m])


def test_active_scene_reference_level():
    x = np.random.default_rng(1).normal(size=(3, 12000))
    result = build_active_scene(x, ARRAY, [-30, 0, 30], source_distances_m=[2, 3, 2])
    np.testing.assert_allclose(result.active_rms, 0.04, atol=1e-14)
    assert result.reference_noise_rms == pytest.approx(0.004)
    np.testing.assert_array_equal(result.scene.relative_delays_samples[:, 7], 0)


def test_experiment_artifact_chain(tmp_path, monkeypatch):
    from pathlib import Path

    from acoustic_array.io.artifacts import load_json, sha256_file, write_json
    from acoustic_array.pipelines import near_field_experiment as experiment

    root = Path(__file__).resolve().parents[1]
    cfg = load_json(root / "configs/experiments/g1_near_field.json")
    cfg.update(
        array_config=str(root / "configs/array/ula16_v1.json"),
        speech_manifest=str(root / "data/manifests/libri_examples.json"),
        duration_s=0.3,
        distances_m=[2],
        adjacent_angles_deg=[30],
        fft_sizes=[512],
    )
    x = np.random.default_rng(2).normal(size=(3, 14400))
    monkeypatch.setattr(experiment, "load_speech_sources", lambda *args: (x, []))
    config = tmp_path / "config.json"
    write_json(config, cfg)
    run = experiment.run_near_field_experiment(config, tmp_path)
    metrics = load_json(run / "metrics.json")
    assert metrics["status"] == "development_only_not_B2"
    assert len(metrics["rows"]) == 4
    for row in metrics["rows"]:
        for band in ["fullband", "evaluation_band"]:
            assert row[band]["failed_targets"] == 0
            expected = min(t["metrics"]["si_sdri_db"] for t in row[band]["targets"])
            assert row[band]["min3_si_sdri_db"] == expected
    for name, digest in load_json(run / "manifest.json")["outputs_sha256"].items():
        assert sha256_file(run / name) == digest
