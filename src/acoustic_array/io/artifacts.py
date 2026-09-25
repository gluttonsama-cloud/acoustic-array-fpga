"""Small explicit artifact writers; no normalization hidden in audio export."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.io import wavfile


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_pcm16(path: Path, sample_rate_hz: int, data: ArrayLike) -> None:
    signal = np.asarray(data, dtype=np.float64)
    if signal.ndim not in (1, 2) or not signal.size or not np.isfinite(signal).all():
        raise ValueError("Expected finite nonempty PCM")
    if np.max(np.abs(signal)) >= 1:
        raise ValueError("Audio export would clip; select an explicit shared recording gain")
    wavfile.write(path, sample_rate_hz, np.rint(signal * 32767).astype(np.int16))
