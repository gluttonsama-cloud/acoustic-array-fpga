"""Attributable three-speaker development comparison with a DOA-bias stress sweep."""

import argparse
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
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
from acoustic_array.io.speech import load_speech_sources
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.free_field import simulate_far_field


def _formatted_metric(group: dict) -> str:
    value = group["si_sdri_db"]
    return f"{value:.3f}" if value is not None else group["metric_status"]["si_sdri_db"]


def _report(run: Path, rows: list[dict], sources: list[dict]) -> None:
    text = [
        "# Three-speaker development comparison",
        "",
        "Real speech excerpts in simulated far field. Development only; NOT B1/B2 acceptance.",
        "One mixture, three known directions, no added room reverberation/channel mismatch. "
        "Offsets are a common bias on all assumed directions, not automatic DOA estimates.",
        "",
        "| FFT | Method | Bias deg | Target deg | SI-SDRi band dB | SI-SDRi full dB |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        text.append(
            f"| {row['fft_size']} | {row['method']} | {row['direction_offset_deg']} | "
            f"{row['true_angle_deg']} | {_formatted_metric(row['evaluation_band'])} | "
            f"{_formatted_metric(row['fullband'])} |"
        )
    text += [
        "",
        "## Attribution",
        "",
        "Source excerpts: CC-BY-4.0. "
        "https://creativecommons.org/licenses/by/4.0/ ; https://www.openslr.org/12/ . "
        "Hosted examples supplied by librosa. Resampled, cropped, level adjusted, "
        "and used to synthesize spatial mixtures and beamformed derivatives.",
        "",
    ]
    text += [f"- {s['attribution']} — {s['url']}" for s in sources]
    text += [
        "",
        "All excerpts are development data, not a held-out set. "
        "Whole-clip RMS equalization and one mixture do not implement formal S1 acceptance. "
        "Offline metric bandpass is identical for reference/input/output and is not "
        "part of the causal beamformer. WAVs use one shared gain.",
        "",
    ]
    (run / "REPORT.md").write_text("\n".join(text), encoding="utf-8")


def run_speech_experiment(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    config = load_json(config_path)
    fields = {
        "schema_version",
        "experiment_id",
        "array_config",
        "speech_manifest",
        "seed",
        "duration_s",
        "source_angles_deg",
        "source_whole_clip_rms",
        "noise_rms",
        "fft_sizes",
        "block_size",
        "direction_offsets_deg",
        "constraint_band_hz",
        "max_condition",
        "min_white_noise_gain_db",
        "evaluation_trim_s",
        "evaluation_band_hz",
        "wav_shared_gain",
    }
    if set(config) != fields or config["schema_version"] != "0.1":
        raise ValueError("Unsupported M2 development config fields/schema")
    array_path = (config_path.parent / config["array_config"]).resolve()
    speech_path = (config_path.parent / config["speech_manifest"]).resolve()
    array_raw = load_json(array_path)
    array = ArrayConfig.from_mapping(array_raw)
    angles = np.asarray(config["source_angles_deg"], dtype=float)
    directions(angles)
    offsets = np.asarray(config["direction_offsets_deg"], dtype=float)
    if offsets.ndim != 1 or not offsets.size or not np.isfinite(offsets).all():
        raise ValueError("Direction offsets must be a finite nonempty vector")
    if len(np.unique(offsets)) != len(offsets) or not np.any(offsets == 0):
        raise ValueError("Use unique direction offsets including zero")
    for offset in offsets:
        directions(angles + offset)
    positive_int(config["block_size"], "block_size")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    duration = finite_real(config["duration_s"], "duration_s")
    if not 0.1 <= duration <= 30:
        raise ValueError("Development runs support 0.1 to 30 seconds")
    gain = finite_real(config["wav_shared_gain"], "wav_shared_gain")
    if not 0 < gain <= 1:
        raise ValueError("wav_shared_gain must be in (0,1]")
    sizes = config["fft_sizes"]
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("fft_sizes must be a nonempty unique list")
    stfts = [STFTConfig(n, n // 2) for n in sizes]
    # Validate all solver combinations before downloading/loading data or creating a run.
    weight_sets = {
        (stft.fft_size, index): constrained_weights(
            array,
            stft,
            angles + offset,
            band_hz=tuple(config["constraint_band_hz"]),
            max_condition=config["max_condition"],
            min_white_noise_gain_db=config["min_white_noise_gain_db"],
        )
        for stft in stfts
        for index, offset in enumerate(offsets)
    }
    trim_s = finite_real(config["evaluation_trim_s"], "evaluation_trim_s")
    trim = round(trim_s * array.sample_rate_hz)
    if trim < max(sizes) or round(duration * array.sample_rate_hz) - 2 * trim < 100:
        raise ValueError("Insufficient trimming/evaluation samples")
    sources, source_records = load_speech_sources(
        speech_path, array.sample_rate_hz, duration, config["source_whole_clip_rms"]
    )
    if len(sources) != len(angles):
        raise ValueError("One direction is required per source file")
    scene = simulate_far_field(sources, array, angles, noise_rms=config["noise_rms"], seed=seed)
    references = scene.components[:, :, array.reference_microphone_index].T
    mixture = scene.mixture[:, array.reference_microphone_index]
    band = tuple(config["evaluation_band_hz"])
    reference_band = bandpass_for_evaluation(references, array.sample_rate_hz, band)
    mixture_band = bandpass_for_evaluation(mixture, array.sample_rate_hz, band)
    expanded = {"experiment": config, "array": array_raw, "speech_sources": source_records}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"speech_{timestamp}_{canonical_hash(expanded)[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.expanded.json", expanded)
    np.savez_compressed(
        run / "input.npz",
        sources=sources,
        mixture=scene.mixture,
        components=scene.components,
        noise=scene.noise,
    )
    write_pcm16(run / "mixture_reference.wav", array.sample_rate_hz, mixture * gain)
    for k in range(len(angles)):
        write_pcm16(run / f"reference_{k}.wav", array.sample_rate_hz, references[:, k] * gain)
    rows = []
    diagnostics = []
    for stft in stfts:
        for offset_index, offset in enumerate(offsets):
            assumed = angles + offset
            constrained = weight_sets[(stft.fft_size, offset_index)]
            weights_filename = f"weights_{stft.fft_size}_offset_index_{offset_index}.npz"
            active = constrained.status != "outside_band_das"
            diagnostics.append(
                {
                    "fft_size": stft.fft_size,
                    "offset_deg": float(offset),
                    "offset_index": offset_index,
                    "weights_file": weights_filename,
                    "active_bin_outputs": int(active.sum()),
                    "fallback_bin_outputs": int(
                        np.sum(np.char.startswith(constrained.status, "fallback"))
                    ),
                    "max_residual": float(np.nanmax(constrained.constraint_residual)),
                    "min_wng_db": float(np.nanmin(constrained.white_noise_gain_db)),
                }
            )
            np.savez_compressed(
                run / weights_filename,
                direction_offset_deg=float(offset),
                assumed_angles_deg=assumed,
                weights=constrained.weights,
                condition=constrained.condition_number,
                residual=constrained.constraint_residual,
                wng_db=constrained.white_noise_gain_db,
                status=constrained.status,
            )
            for method, weights in [
                ("das", das_weights(array, stft, assumed)),
                ("fixed_constraint", constrained.weights),
            ]:
                output = beamform_with_weights(
                    scene.mixture, array, stft, weights, block_size=config["block_size"]
                )
                output_band = bandpass_for_evaluation(output, array.sample_rate_hz, band)
                if offset == 0:
                    np.save(run / f"{method}_{stft.fft_size}.npy", output)
                    for k in range(len(angles)):
                        write_pcm16(
                            run / f"{method}_{stft.fft_size}_target_{k}.wav",
                            array.sample_rate_hz,
                            output[:, k] * gain,
                        )
                for k, angle in enumerate(angles):
                    rows.append(
                        {
                            "fft_size": stft.fft_size,
                            "method": method,
                            "direction_offset_deg": float(offset),
                            "target": k,
                            "true_angle_deg": float(angle),
                            "assumed_angle_deg": float(assumed[k]),
                            "output_peak_abs": float(np.max(np.abs(output[:, k]))),
                            "evaluation_band": comparison_metrics(
                                reference_band[trim:-trim, k],
                                mixture_band[trim:-trim],
                                output_band[trim:-trim, k],
                            ),
                            "fullband": comparison_metrics(
                                references[trim:-trim, k],
                                mixture[trim:-trim],
                                output[trim:-trim, k],
                            ),
                        }
                    )
            print(f"Finished N={stft.fft_size}, DOA bias={offset:g} deg", flush=True)
    write_json(
        run / "metrics.json",
        {
            "status": "development_only_not_B1_or_B2",
            "mode": "known_direction_with_bias_sweep",
            "reference_microphone": array.reference_microphone_index,
            "evaluation_band_hz": list(band),
            "trim_samples_each_end": trim,
            "alignment": "same_sample_indices_no_search",
            "activity_mask": "none_whole_cropped_clip",
            "filter": "6th_order_Butterworth_sosfiltfilt_before_trim",
            "band_outside_constraints": "DAS",
            "wav_shared_gain": gain,
            "rows": rows,
            "constraint_diagnostics": diagnostics,
        },
    )
    _report(run, rows, source_records)
    package_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        p.relative_to(package_root).as_posix(): sha256_file(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    write_json(
        run / "manifest.json",
        {
            "schema_version": "0.1",
            "created_utc": timestamp,
            "config_sha256": canonical_hash(expanded),
            "source_code_files_sha256": code_hashes,
            "source_code_sha256": canonical_hash(code_hashes),
            "input_files_sha256": {
                str(p): sha256_file(p) for p in [config_path, array_path, speech_path]
            },
            "speech_manifest": load_json(speech_path),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "soundfile", "acoustic-array-fpga"]
                },
            },
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "outputs_sha256": {
                p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()
            },
            "limitations": [
                "one_development_mixture",
                "oracle_directions_not_estimation",
                "no_added_reverberation",
                "no_hardware_validation",
                "not_heldout",
            ],
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_speech_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
