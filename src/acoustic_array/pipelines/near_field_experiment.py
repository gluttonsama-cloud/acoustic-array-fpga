"""G1 direct-path development diagnostic with oracle directions/distances.

One three-speaker group reused across conditions, never independent acceptance.
Scene pressure is normalized at MIC reference; range is not a loudness test.
"""

import argparse
import importlib.metadata
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig, finite_real, positive_int
from acoustic_array.evaluation.activity import ActivityConfig, masked_comparison
from acoustic_array.evaluation.metrics import bandpass_for_evaluation
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.io.speech import load_speech_sources
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import build_active_scene


def run_near_field_experiment(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    fields = {
        "schema_version",
        "array_config",
        "speech_manifest",
        "duration_s",
        "distances_m",
        "adjacent_angles_deg",
        "fft_sizes",
        "seed",
        "active_rms",
        "noise_snr_db",
        "trim_samples",
        "block_size",
        "constraint_band_hz",
        "min_wng_db",
        "max_condition",
    }
    if set(cfg) != fields or cfg["schema_version"] != "0.1":
        raise ValueError("Invalid near-field experiment config")
    for key, low, high in [("distances_m", 0.5, 10), ("adjacent_angles_deg", 1, 90)]:
        values = cfg[key]
        if not isinstance(values, list) or not values or len(values) > 10:
            raise ValueError(f"Invalid {key}")
        if any(not low <= finite_real(v, key) <= high for v in values) or len(set(values)) != len(
            values
        ):
            raise ValueError(f"Invalid {key}")
    positive_int(cfg["block_size"], "block_size")
    positive_int(cfg["trim_samples"], "trim_samples")
    if not 0.1 <= finite_real(cfg["duration_s"], "duration_s") <= 30:
        raise ValueError("Invalid duration")
    sizes = cfg["fft_sizes"]
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Invalid FFT list")
    stfts = [STFTConfig(n, n // 2) for n in sizes]
    if cfg["trim_samples"] < max(sizes):
        raise ValueError("Trim must cover largest FFT")
    array_path = (config_path.parent / cfg["array_config"]).resolve()
    speech_path = (config_path.parent / cfg["speech_manifest"]).resolve()
    array_raw = load_json(array_path)
    array = ArrayConfig.from_mapping(array_raw)
    if array.sample_rate_hz <= 10000:
        raise ValueError("Evaluation band requires sample rate > 10000 Hz")
    sources, records = load_speech_sources(
        speech_path, array.sample_rate_hz, cfg["duration_s"], 0.04
    )
    if len(sources) != 3:
        raise ValueError("Diagnostic requires three development sources")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"g1_near_field_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run / "sources.npz", sources=sources)
    expanded = {
        "config": cfg,
        "array": array_raw,
        "speech_sources": records,
        "activity": asdict(ActivityConfig()),
        "evaluation_band_hz": [500, 5000],
        "propagation": "direct_spherical_reference_pressure_no_absolute_delay",
        "independent_speaker_groups": 1,
        "reverberation": False,
    }
    write_json(run / "config.expanded.json", expanded)
    rows = []
    for distance in cfg["distances_m"]:
        for gap in cfg["adjacent_angles_deg"]:
            case = f"r{distance}_gap{gap}"
            angles, distances = [-gap, 0, gap], [distance] * 3
            built = build_active_scene(
                sources,
                array,
                angles,
                source_distances_m=distances,
                active_rms=cfg["active_rms"],
                noise_snr_db=cfg["noise_snr_db"],
                seed=cfg["seed"],
                trim_samples=cfg["trim_samples"],
            )
            np.savez_compressed(
                run / f"{case}_scene.npz",
                masks=built.masks,
                source_gains=built.source_gains,
                delays_samples=built.scene.relative_delays_samples,
            )
            scene = built.scene
            ref = array.reference_microphone_index
            references, mixture = scene.components[:, :, ref].T, scene.mixture[:, ref]
            rb = bandpass_for_evaluation(references, array.sample_rate_hz, (500, 5000))
            mb = bandpass_for_evaluation(mixture, array.sample_rate_hz, (500, 5000))
            crop = slice(cfg["trim_samples"], -cfg["trim_samples"])
            for stft in stfts:
                for model, ranges in [("plane", None), ("spherical", distances)]:
                    constrained = constrained_weights(
                        array,
                        stft,
                        angles,
                        source_distances_m=ranges,
                        band_hz=tuple(cfg["constraint_band_hz"]),
                        max_condition=cfg["max_condition"],
                        min_white_noise_gain_db=cfg["min_wng_db"],
                    )
                    das = das_weights(array, stft, angles, source_distances_m=ranges)
                    np.savez_compressed(
                        run / f"{case}_{stft.fft_size}_{model}_weights.npz",
                        constrained=constrained.weights,
                        das=das,
                        status=constrained.status,
                        condition=constrained.condition_number,
                        residual=constrained.constraint_residual,
                        wng_db=constrained.white_noise_gain_db,
                    )
                    for method, weights in [("das", das), ("constraint", constrained.weights)]:
                        out = beamform_with_weights(
                            scene.mixture, array, stft, weights, block_size=cfg["block_size"]
                        )
                        ob = bandpass_for_evaluation(out, array.sample_rate_hz, (500, 5000))
                        row = {
                            "case": case,
                            "distance_m": distance,
                            "adjacent_angle_deg": gap,
                            "fft_size": stft.fft_size,
                            "model": model,
                            "method": method,
                            "measured_active_snr_db": built.measured_active_snr_db.tolist(),
                        }
                        for label, r, m, e in [
                            ("fullband", references, mixture, out),
                            ("evaluation_band", rb, mb, ob),
                        ]:
                            scores = [
                                masked_comparison(
                                    r[crop, k],
                                    m[crop],
                                    e[crop, k],
                                    built.masks[k, crop],
                                    array.sample_rate_hz,
                                )
                                for k in range(3)
                            ]
                            values = [
                                s["metrics"]["si_sdri_db"] if s["metrics"] else None for s in scores
                            ]
                            row[label] = {
                                "targets": scores,
                                "min3_si_sdri_db": min(values)
                                if all(v is not None for v in values)
                                else None,
                                "failed_targets": sum(v is None for v in values),
                            }
                        rows.append(row)
            print(f"Completed {case}", flush=True)
    write_json(run / "metrics.json", {"status": "development_only_not_B2", "rows": rows})
    package = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "created_utc": stamp,
            "config_sha256": canonical_hash(expanded),
            "argv": sys.argv,
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "soundfile", "acoustic-array-fpga"]
                },
            },
            "input_files_sha256": {
                str(p): sha256_file(p) for p in [config_path, array_path, speech_path]
            },
            "source_code_files_sha256": {
                p.relative_to(package).as_posix(): sha256_file(p) for p in package.rglob("*.py")
            },
            "outputs_sha256": {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()},
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_near_field_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
