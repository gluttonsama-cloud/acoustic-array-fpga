"""Bounded recording replay with explicit geometry, mapping, and calibration."""

import argparse
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_array.beamforming.mpdr import mixture_covariance, mpdr_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import directions
from acoustic_array.dsp.stft import StreamingSTFT
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.pipelines.streaming import beamform_with_weights


def channel_health(pcm: np.ndarray) -> dict:
    """Observable file checks only; this cannot certify physical synchronization."""
    peak = np.max(np.abs(pcm), axis=0)
    duplicates = [
        (a, b)
        for a in range(pcm.shape[1])
        for b in range(a)
        if np.array_equal(pcm[:, a], pcm[:, b])
    ]
    silent = np.flatnonzero(peak == 0).tolist()
    clipped = np.flatnonzero(peak >= 1 - 2**-23).tolist()
    return {
        "peak_by_mic": peak.tolist(),
        "rms_by_mic": np.sqrt(np.mean(pcm**2, axis=0)).tolist(),
        "dc_by_mic": pcm.mean(axis=0).tolist(),
        "zero_channels": silent,
        "identical_channel_pairs": duplicates,
        "full_scale_channels": clipped,
        "physical_sync_verified": False,
        "processing_blocked": bool(silent or duplicates or clipped),
    }


def replay_recording(
    wav: Path,
    profile: Path,
    output_root: Path,
    *,
    calibration_start: float,
    calibration_seconds: float,
    angles: list[float] | None = None,
    source_count: int | None = None,
) -> Path:
    """No source truth, clean audio, or RIR is accepted by the replay API."""
    if (angles is None) == (source_count is None):
        raise ValueError("Choose explicit manual angles OR an explicit source-count prior")
    if angles is not None:
        directions(angles)
        if not 1 <= len(angles) <= 3 or len(set(angles)) != len(angles):
            raise ValueError("One to three distinct manual angles required")
    if source_count is not None and (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or not 1 <= source_count <= 3
    ):
        raise ValueError("source_count must be an integer from one to three")
    if (
        not np.isfinite([calibration_start, calibration_seconds]).all()
        or calibration_start < 0
        or calibration_seconds <= 0
    ):
        raise ValueError("Explicit finite nonnegative start and positive duration required")
    frozen_inputs = {profile.resolve(): sha256_file(profile)}
    meta = load_json(profile)
    if set(meta) != {"array_config", "wav_channel_for_mic", "geometry_evidence"}:
        raise ValueError("Invalid recording profile fields")
    evidence = meta["geometry_evidence"]
    if (
        not isinstance(evidence, dict)
        or evidence.get("kind") not in ("measured", "synthetic")
        or not isinstance(evidence.get("note"), str)
        or not evidence["note"].strip()
    ):
        raise ValueError("Explicit geometry evidence kind and description required")
    array_path = (profile.resolve().parent / meta["array_config"]).resolve()
    frozen_inputs[array_path] = sha256_file(array_path)
    array_raw = load_json(array_path)
    if array_raw.get("geometry", {}).get("kind") != "explicit_coordinates":
        raise ValueError("Explicit microphone coordinates required, no implicit array fallback")
    array = ArrayConfig.from_mapping(array_raw)
    if array.microphone_count != 16 or array.sample_rate_hz != 48000:
        raise ValueError("This first recording adapter requires 16 microphones at 48 kHz")
    mapping = meta["wav_channel_for_mic"]
    if (
        not isinstance(mapping, list)
        or any(type(i) is not int for i in mapping)
        or sorted(mapping) != list(range(16))
    ):
        raise ValueError("Channel mapping must be a permutation of WAV indices 0..15")
    frozen_inputs[wav.resolve()] = sha256_file(wav)

    def verify_inputs() -> None:
        if any(sha256_file(path) != digest for path, digest in frozen_inputs.items()):
            raise ValueError("Input or configuration changed during replay")

    info = sf.info(wav)
    if (
        info.format not in ("WAV", "WAVEX", "RF64")
        or info.channels != 16
        or info.samplerate != 48000
        or info.subtype not in ("PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE")
        or not 0 < info.frames <= 60 * 48000
    ):
        raise ValueError("Require an uncompressed 16-channel 48 kHz WAV of at most 60 seconds")
    start = round(calibration_start * array.sample_rate_hz)
    end = start + round(calibration_seconds * array.sample_rate_hz)
    stft = STFTConfig()
    if end - start < (array.microphone_count + 1) * stft.hop_size or end >= info.frames:
        raise ValueError("Calibration too short or leaves no post-calibration output")
    pcm, read_rate = sf.read(wav, dtype="float64", always_2d=True)
    verify_inputs()
    if pcm.shape != (info.frames, 16) or read_rate != info.samplerate:
        raise ValueError("WAV metadata changed during read")
    if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 2:
        raise ValueError("Nonfinite or invalid normalized PCM amplitude")
    pcm = pcm[:, mapping]
    health = channel_health(pcm)
    prefix = pcm[start:end]
    candidates = []
    output = None
    weights_result = None
    if health["processing_blocked"]:
        status = "reference_only_channel_anomaly"
    elif np.max(np.abs(prefix)) == 0:
        status = "reference_only_silent_calibration"
    else:
        if angles is not None:
            candidates = sorted(map(float, angles))
        else:
            from acoustic_array.localization.music_reference import music_candidates
            from acoustic_array.localization.onset import onset_filter

            frames = StreamingSTFT(stft, 16).push(prefix, sample_start=0)
            spectra = np.stack([f.spectrum for f in frames if f.sample_start >= 0])
            selection = onset_filter(spectra)
            result = music_candidates(
                selection.filtered_spectra,
                array,
                np.fft.rfftfreq(stft.fft_size, 1 / array.sample_rate_hz),
                np.arange(-90.0, 91.0),
                num_sources=source_count,
                frequency_mask=selection.selected_counts > 0,
            )
            candidates = sorted(p.angle_deg for p in result.peaks)
        if not candidates:
            status = "reference_only_no_candidates"
        else:
            covariance, _ = mixture_covariance(prefix, array, stft)
            weights_result = mpdr_weights(covariance, array, stft, candidates)
            tail = beamform_with_weights(pcm[end:], array, stft, weights_result.weights)
            output = np.pad(tail, ((end, 0), (0, 0)))
            status = (
                "manual_direction_replay" if angles is not None else "experimental_candidate_replay"
            )
    verify_inputs()
    run = output_root.resolve() / f"recording_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    run.mkdir(parents=True, exist_ok=False)
    reference = pcm[:, array.reference_microphone_index]
    sf.write(run / "reference_raw.wav", reference, 48000, subtype="FLOAT")
    peak = max(
        float(np.max(np.abs(reference))), 0 if output is None else float(np.max(np.abs(output)))
    )
    gain = min(1.0, 0.95 / peak) if peak else 1.0
    sf.write(run / "reference_listen.wav", reference * gain, 48000, subtype="PCM_24")
    if output is not None:
        np.savez_compressed(
            run / "outputs.npz",
            output=output,
            weights=weights_result.weights,
            weight_status=weights_result.status,
        )
        for k in range(output.shape[1]):
            sf.write(run / f"candidate_port{k}.wav", output[:, k] * gain, 48000, subtype="PCM_24")
    write_json(
        run / "report.json",
        {
            "status": status,
            "profile": meta,
            "array": array_raw,
            "geometry_evidence_is_user_declaration": True,
            "health": health,
            "input_frames": info.frames,
            "sample_rate_hz": 48000,
            "source_count_prior": source_count,
            "processing_settings": {
                "fft_size": 512,
                "hop_size": 256,
                "diagonal_loading": 0.1,
                "beam_band_hz": [100, 5000],
                "max_condition": 1e6,
                "min_wng_db": -10,
                "localization": "onset_normmusic" if source_count is not None else "manual",
                "localization_band_hz": [500, 5000],
                "minimum_peak_separation_deg": 10,
            },
            "calibration_samples": [start, end],
            "output_valid_from_sample": end if output is not None else None,
            "candidate_angles_deg": candidates,
            "candidate_identity_verified": False,
            "listening_gain": gain,
            "metric_policy": "no clean reference: no SI-SDR or separation claims",
            "scope": "bounded file replay, not a live device or FPGA acceptance",
        },
    )
    package = Path(__file__).resolve().parents[1]
    for file in package.rglob("*.py"):
        dest = run / "source_snapshot" / file.relative_to(package)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.read_bytes())
    verify_inputs()
    dependencies = {}
    for package_name in ("numpy", "scipy", "soundfile", "pyroomacoustics"):
        try:
            dependencies[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package_name] = None
    write_json(
        run / "manifest.json",
        {
            "input_sha256": frozen_inputs[wav.resolve()],
            "profile_sha256": frozen_inputs[profile.resolve()],
            "array_sha256": frozen_inputs[array_path],
            "argv": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
            "outputs_sha256": {
                f.relative_to(run).as_posix(): sha256_file(f) for f in run.rglob("*") if f.is_file()
            },
        },
    )
    write_json(
        run / "completion.pending.json", {"manifest_sha256": sha256_file(run / "manifest.json")}
    )
    (run / "completion.pending.json").replace(run / "completion.json")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-start", type=float, required=True)
    parser.add_argument("--calibration-seconds", type=float, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--angles", type=float, nargs="+")
    modes.add_argument("--source-count", type=int)
    args = vars(parser.parse_args())
    print(replay_recording(**args))


if __name__ == "__main__":
    main()
