"""G1 room diagnostic: direct-reference quality and reverberant-target separation.

Fixed oracle direct geometry only. RIRs are retained for reconstruction/decay
checks, never provided to the weight solver. No acceptance or retuning on S1.
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
from acoustic_array.evaluation.decay import decay_summary
from acoustic_array.evaluation.metrics import bandpass_for_evaluation
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.io.speech import load_speech_sources
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomConfig, generate_room_rirs, propagate_room


def _score(references, mixture, output, masks, array, trim) -> dict:
    result = {}
    crop = slice(trim, -trim)
    for label in ["fullband", "evaluation_band"]:
        r, m, e = references, mixture, output
        if label == "evaluation_band":
            r, m, e = [
                bandpass_for_evaluation(x, array.sample_rate_hz, (500, 5000)) for x in [r, m, e]
            ]
        targets = [
            masked_comparison(r[crop, k], m[crop], e[crop, k], masks[k, crop], array.sample_rate_hz)
            for k in range(references.shape[1])
        ]
        values = [t["metrics"]["si_sdri_db"] if t["metrics"] else None for t in targets]
        result[label] = {
            "targets": targets,
            "failed_targets": sum(v is None for v in values),
            "min3_si_sdri_db": min(values) if all(v is not None for v in values) else None,
        }
    return result


def run_room_experiment(config_path: Path, output_root: Path) -> Path:
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    fields = {
        "schema_version",
        "array_config",
        "speech_manifest",
        "duration_s",
        "distances_m",
        "adjacent_angles_deg",
        "rt60_targets_s",
        "room",
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
        raise ValueError("Invalid room config schema")
    for key, low, high in [
        ("distances_m", 0.5, 10),
        ("adjacent_angles_deg", 1, 90),
        ("rt60_targets_s", 0, 1),
    ]:
        values = cfg[key]
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 10
            or any(not low <= finite_real(v, key) <= high for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid {key}")
    for key in ["block_size", "trim_samples"]:
        positive_int(cfg[key], key)
    if not 0.1 <= finite_real(cfg["duration_s"], "duration_s") <= 30:
        raise ValueError("Invalid duration")
    sizes = cfg["fft_sizes"]
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Invalid FFT sizes")
    stfts = [STFTConfig(n, n // 2) for n in sizes]
    room = RoomConfig(**cfg["room"])
    array_path = (config_path.parent / cfg["array_config"]).resolve()
    speech_path = (config_path.parent / cfg["speech_manifest"]).resolve()
    array_raw = load_json(array_path)
    array = ArrayConfig.from_mapping(array_raw)
    if array.sample_rate_hz <= 10000:
        raise ValueError("Evaluation band requires fs > 10000")
    if cfg["trim_samples"] < max(max(sizes), round(room.rir_duration_s * array.sample_rate_hz)):
        raise ValueError("Trim must cover maximum RIR duration and FFT")
    sources, records = load_speech_sources(
        speech_path, array.sample_rate_hz, cfg["duration_s"], 0.04
    )
    if len(sources) != 3 or sources.shape[1] <= 2 * cfg["trim_samples"]:
        raise ValueError("Need three sufficiently long development sources")
    sources = sources - sources.mean(axis=1, keepdims=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"g1_room_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run / "sources.npz", sources=sources)
    expanded = {
        "config": cfg,
        "array": array_raw,
        "speech_sources": records,
        "activity": asdict(ActivityConfig()),
        "evaluation_band_hz": [500, 5000],
        "normalization": "direct MIC reference active RMS; reverberant components share gain",
        "references": ["direct", "reverberant_target_at_reference_mic"],
        "independent_speaker_groups": 1,
        "rir_generator": "0.3.0",
        "rir_settings": {
            "hp_filter": False,
            "order": -1,
            "dim": 3,
            "duration_rule": "positive RT: min(max_duration, 1.5*RT+0.05); zero: max_duration",
        },
    }
    write_json(run / "config.expanded.json", expanded)
    package = Path(__file__).resolve().parents[1]
    source_hashes = {
        p.relative_to(package).as_posix(): sha256_file(p) for p in package.rglob("*.py")
    }
    # Snapshot exact source before a long run, so later maintenance cannot rewrite provenance.
    for name in source_hashes:
        target = run / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((package / name).read_bytes())
    rows, diagnostics = [], []
    expected_cases = (
        len(cfg["distances_m"]) * len(cfg["adjacent_angles_deg"]) * len(cfg["rt60_targets_s"])
    )

    def checkpoint(run_status: str) -> None:
        write_json(
            run / "metrics.json",
            {
                "status": "development_only_not_B2",
                "run_status": run_status,
                "expected_cases": expected_cases,
                "completed_cases": len(diagnostics),
                "rows": rows,
            },
        )
        write_json(run / "room_diagnostics.json", diagnostics)

    checkpoint("in_progress")
    for distance in cfg["distances_m"]:
        for gap in cfg["adjacent_angles_deg"]:
            angles, distances = [-gap, 0, gap], [distance] * 3
            weight_sets = []
            for stft in stfts:
                constrained = constrained_weights(
                    array,
                    stft,
                    angles,
                    source_distances_m=distances,
                    band_hz=tuple(cfg["constraint_band_hz"]),
                    max_condition=cfg["max_condition"],
                    min_white_noise_gain_db=cfg["min_wng_db"],
                )
                das = das_weights(array, stft, angles, source_distances_m=distances)
                weight_sets.append((stft, constrained.weights, das))
                np.savez_compressed(
                    run / f"r{distance}_gap{gap}_{stft.fft_size}_weights.npz",
                    constrained=constrained.weights,
                    das=das,
                    status=constrained.status,
                    residual=constrained.constraint_residual,
                    condition=constrained.condition_number,
                    wng_db=constrained.white_noise_gain_db,
                )
            for rt in cfg["rt60_targets_s"]:
                case = f"r{distance}_gap{gap}_rt{rt}"
                print(f"Generating {case}", flush=True)
                rirs = generate_room_rirs(array, angles, distances, room, rt)
                base, direct = propagate_room(sources, array, rirs)
                built = normalize_active_scene(
                    base,
                    array,
                    activity_components=direct,
                    active_rms=cfg["active_rms"],
                    noise_snr_db=cfg["noise_snr_db"],
                    seed=cfg["seed"],
                    trim_samples=cfg["trim_samples"],
                )
                scene = built.scene
                ref = array.reference_microphone_index
                references = direct[:, :, ref].T * built.source_gains
                reverberant_refs = scene.components[:, :, ref].T
                np.savez_compressed(
                    run / f"{case}_reconstruction.npz",
                    rirs=rirs.impulse_responses,
                    direct_rirs=rirs.direct_responses,
                    sources_m=rirs.source_positions_m,
                    microphones_m=rirs.microphone_positions_m,
                    direct_distances_m=rirs.direct_distances_m,
                    masks=built.masks,
                    gains=built.source_gains,
                )
                decay = []
                if rt:
                    for k in range(3):
                        for mic in range(array.microphone_count):
                            h = rirs.impulse_responses[k, :, mic]
                            decay.append(
                                {
                                    "source": k,
                                    "mic": mic,
                                    "broadband": decay_summary(h, array.sample_rate_hz),
                                    "evaluation_band": decay_summary(
                                        bandpass_for_evaluation(
                                            h, array.sample_rate_hz, (500, 5000)
                                        ),
                                        array.sample_rate_hz,
                                    ),
                                }
                            )
                diagnostics.append(
                    {
                        "case": case,
                        "target_rt60_s": rt,
                        "energy_absorption": rirs.energy_absorption,
                        "actual_rir_duration_s": rirs.impulse_responses.shape[1]
                        / array.sample_rate_hz,
                        "direct_active_rms": built.active_rms.tolist(),
                        "direct_active_snr_db": built.measured_active_snr_db.tolist(),
                        "decay": decay,
                    }
                )
                for stft, constraint, das in weight_sets:
                    for method, weights in [("das", das), ("constraint", constraint)]:
                        output = beamform_with_weights(
                            scene.mixture, array, stft, weights, block_size=cfg["block_size"]
                        )
                        row = {
                            "case": case,
                            "distance_m": distance,
                            "adjacent_angle_deg": gap,
                            "target_rt60_s": rt,
                            "fft_size": stft.fft_size,
                            "method": method,
                        }
                        for label, reference in [
                            ("direct_reference", references),
                            ("reverberant_reference", reverberant_refs),
                        ]:
                            row[label] = _score(
                                reference,
                                scene.mixture[:, ref],
                                output,
                                built.masks,
                                array,
                                cfg["trim_samples"],
                            )
                        rows.append(row)
                checkpoint("in_progress")
                print(f"Completed {case}", flush=True)
    checkpoint("complete")
    write_json(
        run / "manifest.json",
        {
            "created_utc": stamp,
            "config_sha256": canonical_hash(expanded),
            "argv": sys.argv,
            "source_code_files_sha256": source_hashes,
            "input_files_sha256": {
                str(p): sha256_file(p) for p in [config_path, array_path, speech_path]
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in [
                        "numpy",
                        "scipy",
                        "soundfile",
                        "rir-generator",
                        "acoustic-array-fpga",
                    ]
                },
            },
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()
            },
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_room_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
