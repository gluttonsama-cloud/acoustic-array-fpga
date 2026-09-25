"""Locally cached, checksum-verified speech with explicit preprocessing metadata."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from acoustic_array.core.config import finite_real, positive_int
from acoustic_array.io.artifacts import load_json, sha256_file


def load_speech_sources(
    manifest_path: Path, sample_rate_hz: int, duration_s: float, target_rms: float = 0.04
) -> tuple[np.ndarray, list[dict]]:
    """Read first duration_s of each mono file, resample and equalize whole-clip RMS.

    No network, repetition, artificial padding, or active-speech level claims.
    Whole-clip RMS here is a development convention, not formal S1 active RMS.
    """
    positive_int(sample_rate_hz, "sample_rate_hz")
    duration = finite_real(duration_s, "duration_s")
    rms = finite_real(target_rms, "target_rms")
    if duration <= 0 or not 0 < rms < 1:
        raise ValueError("duration and target RMS must be positive (RMS < 1)")
    manifest = load_json(manifest_path)
    if manifest.get("license") != "CC-BY-4.0" or not manifest.get("files"):
        raise ValueError("Expected attributed CC-BY-4.0 development data")
    sources = []
    records = []
    count = round(duration * sample_rate_hz)
    for entry in manifest["files"]:
        path = (manifest_path.parent / entry["path"]).resolve()
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Speech checksum mismatch: {path}")
        pcm, original_rate = sf.read(path, dtype="float64", always_2d=True)
        if pcm.shape[1] != 1 or not np.isfinite(pcm).all():
            raise ValueError("Expected finite mono speech")
        if len(pcm) / original_rate < duration:
            raise ValueError("Speech is too short; no repetition/padding is allowed")
        divisor = gcd(sample_rate_hz, original_rate)
        # Resample the whole utterance before cropping; record this boundary convention.
        audio = resample_poly(pcm[:, 0], sample_rate_hz // divisor, original_rate // divisor)
        audio = audio[:count].copy()
        audio -= audio.mean()
        original_rms = float(np.sqrt(np.mean(audio**2)))
        if original_rms < 1e-12:
            raise ValueError("Silent source")
        gain = rms / original_rms
        audio *= gain
        sources.append(audio)
        records.append(
            {
                **entry,
                "original_sample_rate_hz": original_rate,
                "original_samples": len(pcm),
                "output_samples": count,
                "crop_start_s": 0,
                "crop_duration_s": duration,
                "processing": "resample_poly_default_kaiser_then_crop_demean_whole_clip_RMS",
                "resample_up": sample_rate_hz // divisor,
                "resample_down": original_rate // divisor,
                "gain": gain,
                "actual_rms": float(np.sqrt(np.mean(audio**2))),
            }
        )
    return np.stack(sources), records
