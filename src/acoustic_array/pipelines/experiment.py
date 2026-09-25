"""M1 reproducible engineering experiment; this is not a speech benchmark."""

import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real, positive_int
from acoustic_array.core.geometry import directions
from acoustic_array.evaluation.metrics import bandpass_for_evaluation, comparison_metrics
from acoustic_array.io.artifacts import (
    canonical_hash,
    load_json,
    sha256_file,
    write_json,
    write_pcm16,
)
from acoustic_array.pipelines.streaming import beamform_pcm
from acoustic_array.simulation.free_field import simulate_far_field, synthetic_sources


def run_experiment(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    config = load_json(config_path)
    expected = {
        "schema_version",
        "experiment_id",
        "array_config",
        "seed",
        "duration_s",
        "source_angles_deg",
        "source_kind",
        "source_band_hz",
        "noise_rms",
        "fft_sizes",
        "block_size",
        "evaluation_trim_s",
        "evaluation_band_hz",
    }
    if set(config) != expected or config["schema_version"] != "0.1":
        raise ValueError("Experiment fields or schema_version do not match M1 schema")
    if config["source_kind"] != "amplitude_modulated_bandlimited_noise_not_speech":
        raise ValueError("Only synthetic engineering inputs are implemented in M1")
    array_path = (config_path.parent / config["array_config"]).resolve()
    array_raw = load_json(array_path)
    array = ArrayConfig.from_mapping(array_raw)
    duration = finite_real(config["duration_s"], "duration_s")
    if not 0.1 <= duration <= 60:
        raise ValueError("M1 in-memory engineering runs support 0.1 to 60 seconds")
    positive_int(config["block_size"], "block_size")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    angles = config["source_angles_deg"]
    directions(angles)
    if len(angles) > array.microphone_count:
        raise ValueError("More outputs than microphones are not supported in M1")
    fft_sizes = config["fft_sizes"]
    if not isinstance(fft_sizes, list) or not fft_sizes or len(set(fft_sizes)) != len(fft_sizes):
        raise ValueError("fft_sizes must be a nonempty unique list")
    stfts = [STFTConfig(n, n // 2) for n in fft_sizes]
    samples = round(duration * array.sample_rate_hz)
    trim_s = finite_real(config["evaluation_trim_s"], "evaluation_trim_s")
    trim = round(trim_s * array.sample_rate_hz)
    if trim < max(fft_sizes) or samples - 2 * trim < 100:
        raise ValueError("Evaluation requires >= one FFT of trim and >=100 remaining samples")
    sources = synthetic_sources(
        array.sample_rate_hz, samples, len(angles), seed, tuple(config["source_band_hz"])
    )
    scene = simulate_far_field(sources, array, angles, noise_rms=config["noise_rms"], seed=seed + 1)
    expanded = {"experiment": config, "array": array_raw}
    config_hash = canonical_hash(expanded)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"{stamp}_{config_hash[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.expanded.json", expanded)
    np.savez_compressed(
        run / "input.npz",
        sources=sources,
        mixture=scene.mixture,
        components=scene.components,
        noise=scene.noise,
        relative_delays_samples=scene.relative_delays_samples,
    )
    write_pcm16(run / "mixture_16ch.wav", array.sample_rate_hz, scene.mixture)
    reference_mix = scene.mixture[:, array.reference_microphone_index]
    write_pcm16(run / "mixture_reference.wav", array.sample_rate_hz, reference_mix)
    rows: list[dict[str, Any]] = []
    band = tuple(config["evaluation_band_hz"])
    references = scene.components[:, :, array.reference_microphone_index].T
    filtered_reference = bandpass_for_evaluation(references, array.sample_rate_hz, band)
    filtered_mix = bandpass_for_evaluation(reference_mix, array.sample_rate_hz, band)
    for stft in stfts:
        output = beamform_pcm(scene.mixture, array, stft, angles, block_size=config["block_size"])
        if output.shape != (samples, len(angles)):
            raise RuntimeError("Incorrect output shape")
        np.save(run / f"das_{stft.fft_size}.npy", output)
        filtered_output = bandpass_for_evaluation(output, array.sample_rate_hz, band)
        for k, angle in enumerate(angles):
            write_pcm16(
                run / f"das_{stft.fft_size}_target_{k}.wav", array.sample_rate_hz, output[:, k]
            )
            row: dict[str, Any] = {"fft_size": stft.fft_size, "target": k, "angle_deg": angle}
            for label, ref, mix, est in [
                ("fullband", references[:, k], reference_mix, output[:, k]),
                ("evaluation_band", filtered_reference[:, k], filtered_mix, filtered_output[:, k]),
            ]:
                row[label] = comparison_metrics(ref[trim:-trim], mix[trim:-trim], est[trim:-trim])
            rows.append(row)
    write_json(
        run / "metrics.json",
        {
            "status": "engineering_only_not_B1_or_B2",
            "mode": "known_direction_oracle",
            "source_kind": config["source_kind"],
            "reference_microphone": array.reference_microphone_index,
            "alignment": "same_sample_indices_no_search",
            "trim_samples_each_end": trim,
            "evaluation_filter": "6th_order_Butterworth_bandpass_sosfiltfilt_before_trim",
            "output_band_policy": "fullband_DAS_no_band_mask_DC_Nyquist_real_projection",
            "evaluation_band_hz": list(band),
            "rows": rows,
        },
    )
    report = [
        "# M1 DAS engineering run",
        "",
        "Synthetic modulated noise, NOT speech. "
        "Known directions, no reverberation or channel errors. Not B1/B2 acceptance.",
        "",
        "| FFT | Target angle | Input SI-SDR (band) | Output SI-SDR (band) | SI-SDRi (band) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        metrics = row["evaluation_band"]
        cells = [
            f"{metrics[key]:.3f}" if metrics[key] is not None else metrics["metric_status"][key]
            for key in ("input_si_sdr_db", "output_si_sdr_db", "si_sdri_db")
        ]
        report.append(f"| {row['fft_size']} | {row['angle_deg']} | " + " | ".join(cells) + " |")
    report += [
        "",
        "Audio files use a shared fixed gain (1.0), PCM16 export; metrics use float64. "
        "No per-output gain normalization, delay search or source permutation.",
        "",
        "See config.expanded.json, metrics.json and manifest.json for definitions and hashes.",
    ]
    (run / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    package_root = Path(__file__).resolve().parents[1]
    code_files = {
        p.relative_to(package_root).as_posix(): sha256_file(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    outputs = {p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()}
    write_json(
        run / "manifest.json",
        {
            "schema_version": "0.1",
            "created_utc": stamp,
            "config_sha256": config_hash,
            "source_code_files_sha256": code_files,
            "source_code_sha256": canonical_hash(code_files),
            "config_input_files": {
                str(config_path): sha256_file(config_path),
                str(array_path): sha256_file(array_path),
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "acoustic-array-fpga")
                },
            },
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "outputs_sha256": outputs,
            "input_data": "input.npz: source/sample/microphone components; no external dataset",
            "limitations": [
                "synthetic_non_speech",
                "far_field_only",
                "oracle_angles",
                "no_hardware_validation",
                "not_statistical_acceptance",
            ],
        },
    )
    return run
