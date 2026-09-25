"""Frozen W1/W2 wood-knock proxy experiment with physical component replay."""

import argparse
import json
import platform
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly

from acoustic_array.beamforming.das import das_weights
from acoustic_array.beamforming.mpdr import mixture_covariance, mpdr_weights
from acoustic_array.beamforming.sector_lcmv import sector_lcmv_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.evaluation.activity import reference_activity
from acoustic_array.evaluation.metrics import bandpass_for_evaluation
from acoustic_array.evaluation.room_components import target_component_summary
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.pipelines.room_component_replay import verified_manifest
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.room import RoomRIR, propagate_room
from acoustic_array.simulation.weak_scene import calibrate_weak_scene


def load_inputs(config_path: Path) -> tuple[dict, Path, dict, ArrayConfig, np.ndarray, np.ndarray]:
    """Verify immutable inputs before creating a run."""
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    if (
        cfg.get("schema_version") != "0.1"
        or cfg.get("cases") != ["r2_gap30_rt0", "r2_gap30_rt0.4"]
        or cfg.get("angle_errors_deg") != [-2, 0, 2]
        or cfg.get("target_source_index") != 1
        or cfg.get("interference_source_index") != 0
        or cfg.get("target_start_s") != 4
        or cfg.get("calibration_seconds") != 3
        or cfg.get("fft_size") != 512
        or cfg.get("band_hz") != [100, 5000]
        or cfg.get("sector_half_width_deg") != 2
        or cfg.get("diagonal_loading") != 0.1
    ):
        raise ValueError("Expected frozen W1/W2 two-case config")
    room = (config_path.parent / cfg["room_run"]).resolve()
    verified_manifest(room)
    parent = load_json(room / "config.expanded.json")
    array = ArrayConfig.from_mapping(parent["array"])
    if array.sample_rate_hz != 48000 or array.microphone_count < 3:
        raise ValueError("Unexpected parent array for W1/W2")
    manifest_path = (config_path.parent / cfg["knock_manifest"]).resolve()
    manifest = load_json(manifest_path)
    knock_path = (manifest_path.parent / manifest["file"]["path"]).resolve()
    if sha256_file(knock_path) != manifest["file"]["sha256"]:
        raise ValueError("Knock file SHA256 mismatch")
    knock, rate = sf.read(knock_path, dtype="float64")
    if rate != 44100 or knock.ndim != 1 or len(knock) != 5 * rate:
        raise ValueError("Expected five-second mono 44.1 kHz ESC-50 recording")
    knock = resample_poly(knock, array.sample_rate_hz, rate)
    with np.load(room / "sources.npz") as z:
        sources = z["sources"].copy()
    if sources.shape[0] != 3 or sources.shape[1] != 10 * array.sample_rate_hz:
        raise ValueError("Expected three parent ten-second sources")
    if len(knock) != 5 * array.sample_rate_hz:
        raise ValueError("Unexpected target timing")
    for case in cfg["cases"]:
        if not (room / f"{case}_reconstruction.npz").is_file():
            raise ValueError(f"Missing verified RIR case {case}")
    return cfg, room, manifest, array, sources, knock


def scene_for_case(
    cfg: dict, room: Path, case: str, array: ArrayConfig, sources: np.ndarray, knock: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    diagnosis = {d["case"]: d for d in load_json_list(room / "room_diagnostics.json")}
    record = diagnosis[case]
    with np.load(room / f"{case}_reconstruction.npz") as z:
        rirs = RoomRIR(
            z["rirs"],
            z["direct_rirs"],
            z["sources_m"],
            z["microphones_m"],
            z["direct_distances_m"],
            record["target_rt60_s"],
            record["energy_absorption"],
        )
        displacement = z["sources_m"][1] - z["microphones_m"].mean(axis=0)
        expected_distance = float(np.linalg.norm(displacement))
    if not np.allclose(displacement, [0.0, 2.0, 0.0], atol=1e-6):
        raise ValueError("Unexpected zero-degree target source geometry")
    signals = np.zeros_like(sources)
    signals[0] = sources[0]
    start = round(cfg["target_start_s"] * array.sample_rate_hz)
    signals[1, start : start + len(knock)] = knock
    scene, direct = propagate_room(signals, array, rirs)
    target = scene.components[1].copy()
    direct_target = direct[1].copy()
    # FFT convolution of a delayed source has numerical pre-echo; preserve physical silence.
    target[:start] = 0
    direct_target[:start] = 0
    mask = reference_activity(
        direct_target[:, array.reference_microphone_index], array.sample_rate_hz
    )
    mask[: round(3.65 * array.sample_rate_hz)] = False
    mask[round(9.35 * array.sample_rate_hz) :] = False
    if int(mask.sum()) < array.sample_rate_hz // 20:
        raise ValueError("Too little target event activity")
    calibrated = calibrate_weak_scene(
        target,
        scene.components[0],
        mask,
        reference_microphone=array.reference_microphone_index,
        target_rms=cfg["target_rms"],
        sir_db=cfg["sir_db"],
    )
    return (
        calibrated.target,
        calibrated.interference,
        direct_target * calibrated.target_gain,
        {
            "mask": mask,
            "target_distance_m": expected_distance,
            "target_gain": calibrated.target_gain,
            "interference_gain": calibrated.interference_gain,
            "calibrated_interference_on_sir_db": calibrated.measured_sir_db,
            "target_prefix_peak": float(np.max(np.abs(calibrated.target[:start]))),
        },
    )


def load_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Expected JSON list")
    return value


def fixed_noise(
    target: np.ndarray, mask: np.ndarray, ref: int, snr_db: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(target.shape)
    target_power = float(np.mean(target[mask, ref] ** 2))
    noise_power = float(np.mean(noise[mask, ref] ** 2))
    return noise * np.sqrt(target_power / (noise_power * 10 ** (snr_db / 10)))


def replay(
    data: np.ndarray, calibration: int, array: ArrayConfig, stft: STFTConfig, weights: np.ndarray
) -> np.ndarray:
    tail = beamform_with_weights(data[calibration:], array, stft, weights)
    return np.pad(tail[:, 0], (calibration, 0))


def score_components(
    reference: np.ndarray,
    direct: np.ndarray,
    target: np.ndarray,
    interference: np.ndarray,
    noise: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    sample_rate: int,
    band: tuple[float, float] | None,
) -> dict:
    vectors = [reference, direct, target, interference, noise, baseline]
    if band is not None:
        vectors = [bandpass_for_evaluation(x, sample_rate, band) for x in vectors]
    return target_component_summary(*(x[mask] for x in vectors))


def run_weak_knock(config_path: Path, output_root: Path) -> Path:
    cfg, room, knock_manifest, array, sources, knock = load_inputs(config_path)
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    calibration = round(cfg["calibration_seconds"] * array.sample_rate_hz)
    if calibration >= round(cfg["target_start_s"] * array.sample_rate_hz):
        raise ValueError("Calibration overlaps target")
    run = output_root.resolve() / ("w1_knock_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ"))
    run.mkdir(parents=True, exist_ok=False)
    expanded = {
        "config": cfg,
        "array": load_json(room / "config.expanded.json")["array"],
        "parent_room_manifest_sha256": sha256_file(room / "manifest.json"),
        "knock_manifest": knock_manifest,
        "knock_manifest_sha256": sha256_file(
            (config_path.resolve().parent / cfg["knock_manifest"]).resolve()
        ),
        "calibration_interval_samples": [0, calibration],
        "target_activity_crop_s": [3.65, 9.35],
        "evaluation_reference": "direct target at parent reference microphone; no alignment search",
        "training": "mixture-only target-absent prefix; frozen weights after sample 3s",
        "comparison": "single output target at 0 degrees; full target includes reflections",
    }
    write_json(run / "config.expanded.json", expanded)
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for path in package.rglob("*.py"):
        name = path.relative_to(package).as_posix()
        dest = run / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[name] = sha256_file(dest)
    rows: list[dict] = []
    expected = len(cfg["cases"]) * 2 * len(cfg["angle_errors_deg"]) * 3

    def checkpoint() -> None:
        write_json(
            run / "metrics.json",
            {
                "status": "development_only_not_hardware_or_rescue_evidence",
                "run_status": "results_complete" if len(rows) == expected else "in_progress",
                "expected_rows": expected,
                "completed_rows": len(rows),
                "rows": rows,
            },
        )

    checkpoint()
    for case_index, case in enumerate(cfg["cases"]):
        target, interference, direct, meta = scene_for_case(cfg, room, case, array, sources, knock)
        mask = meta.pop("mask")
        np.save(run / f"{case}_activity_mask.npy", mask)
        ref = array.reference_microphone_index
        noise = fixed_noise(target, mask, ref, cfg["noise_snr_db"], cfg["seed"] + case_index)
        case_audio: dict[str, np.ndarray] = {}
        for enabled in (True, False):
            setting = "interference_on" if enabled else "interference_off"
            other = interference if enabled else np.zeros_like(interference)
            mix = target + other + noise
            if np.max(np.abs(target[:calibration])) > 1e-12:
                raise ValueError("Target leaks into calibration")
            covariance, frames = mixture_covariance(mix[:calibration], array, stft)
            np.savez_compressed(
                run / f"{case}_{setting}_calibration.npz",
                covariance=covariance,
                frames=frames,
            )
            for name, data in (
                ("target", target[:, ref]),
                ("interference", other[:, ref]),
                ("noise", noise[:, ref]),
                ("mixture_reference", mix[:, ref]),
                ("direct_reference", direct[:, ref]),
            ):
                case_audio[f"{setting}_{name}"] = data
            for error in cfg["angle_errors_deg"]:
                name_prefix = f"{setting}_error{error:+g}"
                direction = [float(error)]
                distance = [meta["target_distance_m"]]
                das = das_weights(array, stft, direction, source_distances_m=distance)
                loaded = mpdr_weights(
                    covariance,
                    array,
                    stft,
                    direction,
                    source_distances_m=distance,
                    diagonal_loading=cfg["diagonal_loading"],
                    band_hz=tuple(cfg["band_hz"]),
                    max_condition=cfg["max_condition"],
                    min_white_noise_gain_db=cfg["min_wng_db"],
                )
                sector = sector_lcmv_weights(
                    covariance,
                    array,
                    stft,
                    direction,
                    source_distances_m=distance,
                    half_width_deg=cfg["sector_half_width_deg"],
                    diagonal_loading=cfg["diagonal_loading"],
                    band_hz=tuple(cfg["band_hz"]),
                    max_condition=cfg["max_condition"],
                    min_white_noise_gain_db=cfg["min_wng_db"],
                )
                methods = (
                    ("das", das, None),
                    ("loaded_single", loaded.weights, loaded),
                    ("sector_lcmv", sector.weights, sector),
                )
                frequencies = np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz)
                in_band = (frequencies >= cfg["band_hz"][0]) & (frequencies <= cfg["band_hz"][1])
                for method, weights, result in methods:
                    key = f"{name_prefix}_{method}"
                    if result is None:
                        np.savez_compressed(run / f"{case}_{key}_weights.npz", weights=weights)
                        diagnostics = {
                            "status_counts": {"das": stft.fft_size // 2 + 1},
                            "in_band_bin_count": int(in_band.sum()),
                            "in_band_status_counts": {"das": int(in_band.sum())},
                        }
                    else:
                        np.savez_compressed(
                            run / f"{case}_{key}_weights.npz",
                            weights=weights,
                            status=result.status,
                            condition=result.condition_number,
                            residual=result.constraint_residual,
                            wng_db=result.white_noise_gain_db,
                        )
                        diagnostics = {
                            "status_counts": dict(Counter(result.status[:, 0].tolist())),
                            "in_band_bin_count": int(in_band.sum()),
                            "in_band_status_counts": dict(
                                Counter(result.status[in_band, 0].tolist())
                            ),
                            "max_finite_condition": float(
                                np.max(
                                    result.condition_number[np.isfinite(result.condition_number)]
                                )
                            )
                            if np.isfinite(result.condition_number).any()
                            else None,
                            "max_finite_residual": float(
                                np.max(
                                    result.constraint_residual[
                                        np.isfinite(result.constraint_residual)
                                    ]
                                )
                            )
                            if np.isfinite(result.constraint_residual).any()
                            else None,
                        }
                    outputs = [
                        replay(x, calibration, array, stft, weights)
                        for x in (target, other, noise, direct, mix)
                    ]
                    t, i, n, d, y = outputs
                    linearity = float(np.max(np.abs(y - t - i - n)))
                    if linearity > 1e-9:
                        raise ValueError(f"Component replay linearity failure: {linearity}")
                    case_audio[key] = y
                    for component_name, component in (
                        ("target", t),
                        ("interference", i),
                        ("noise", n),
                        ("direct_target", d),
                    ):
                        case_audio[f"{key}_{component_name}"] = component
                    reference = direct[:, ref]
                    baseline = mix[:, ref]
                    rows.append(
                        {
                            "case": case,
                            "interference": setting,
                            "angle_error_deg": error,
                            "method": method,
                            "calibration_frames": frames,
                            "target_activity_samples": int(mask.sum()),
                            "scene": {
                                **meta,
                                "actual_interference_sir": (
                                    {
                                        "db": meta["calibrated_interference_on_sir_db"],
                                        "status": "finite",
                                    }
                                    if enabled
                                    else {"db": None, "status": "positive_infinity"}
                                ),
                                "noise_snr_db": float(
                                    10
                                    * np.log10(
                                        np.mean(target[mask, ref] ** 2)
                                        / np.mean(noise[mask, ref] ** 2)
                                    )
                                ),
                            },
                            "weights": diagnostics,
                            "linearity_max_abs_error": linearity,
                            "fullband": score_components(
                                reference, d, t, i, n, baseline, mask, array.sample_rate_hz, None
                            ),
                            "evaluation_band": score_components(
                                reference,
                                d,
                                t,
                                i,
                                n,
                                baseline,
                                mask,
                                array.sample_rate_hz,
                                (500, cfg["band_hz"][1]),
                            ),
                        }
                    )
                    checkpoint()
                    print(f"Completed {case} {key} ({len(rows)}/{expected})", flush=True)
        peak = max(float(np.max(np.abs(x))) for x in case_audio.values())
        gain = min(1.0, 0.9 / peak) if peak else 1.0
        write_json(
            run / f"{case}_audio_gain.json",
            {
                "common_gain": gain,
                "pre_gain_peak": peak,
                "format": "WAV float32",
                "policy": (
                    "same gain for all conditions and methods within case; "
                    "no individual normalization"
                ),
            },
        )
        for key, data in case_audio.items():
            sf.write(
                run / f"{case}_{key}.wav",
                (data * gain).astype(np.float32),
                array.sample_rate_hz,
                subtype="FLOAT",
            )
    checkpoint()
    write_json(
        run / "manifest.json",
        {
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "rir_generator": version("rir-generator"),
            "config_sha256": canonical_hash(expanded),
            "input_config_sha256": sha256_file(config_path),
            "knock_sha256": knock_manifest["file"]["sha256"],
            "source_code_files_sha256": hashes,
            "outputs_sha256": {
                p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob("*") if p.is_file()
            },
        },
    )
    pending = run / "completion.pending.json"
    write_json(
        pending,
        {
            "run_status": "complete",
            "manifest_sha256": sha256_file(run / "manifest.json"),
        },
    )
    pending.replace(run / "completion.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/w1_knock.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_weak_knock(args.config, args.output_root))


if __name__ == "__main__":
    main()
