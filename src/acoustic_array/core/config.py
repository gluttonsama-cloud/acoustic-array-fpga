"""Configuration validation without filesystem or application dependencies."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class ArrayConfig:
    sample_rate_hz: int
    microphone_count: int
    spacing_m: float
    reference_microphone_index: int
    sound_speed_m_per_s: float = 343.0
    positions_m: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        positive_int(self.sample_rate_hz, "sample_rate_hz")
        positive_int(self.microphone_count, "microphone_count")
        for name in ("spacing_m", "sound_speed_m_per_s"):
            if finite_real(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        index = self.reference_microphone_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("reference_microphone_index must be an integer")
        if not 0 <= index < self.microphone_count:
            raise ValueError("reference_microphone_index is out of range")
        if self.positions_m is not None:
            raw = np.asarray(self.positions_m, dtype=object)
            if raw.shape != (self.microphone_count, 3):
                raise ValueError("positions_m must have shape [microphone_count,3]")
            values = np.array([finite_real(v, "position") for v in raw.flat]).reshape(raw.shape)
            if np.max(np.abs(values)) > 100:
                raise ValueError("Local microphone coordinates must be within 100 metres")
            if not np.allclose(values.mean(axis=0), 0, rtol=0, atol=1e-10):
                raise ValueError("positions_m origin must be microphone centroid")
            separations = np.linalg.norm(values[:, None] - values[None, :], axis=2)
            np.fill_diagonal(separations, np.inf)
            if np.any(separations < 1e-6):
                raise ValueError("Microphones must be at least one micrometre apart")
            object.__setattr__(
                self, "positions_m", tuple(tuple(float(v) for v in row) for row in values)
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ArrayConfig":
        required = (
            "sample_rate_hz",
            "microphone_count",
            "spacing_m",
            "reference_microphone_index",
            "sound_speed_m_per_s",
        )
        missing = set(required) - raw.keys()
        if missing:
            raise ValueError(f"Missing array fields: {sorted(missing)}")
        explicit = raw.get("schema_version") == "0.2"
        if raw.get("schema_version") not in {"0.1", "0.2"}:
            raise ValueError("Unsupported array schema_version")
        if not explicit and "positions_m" in raw:
            raise ValueError("Explicit positions require schema_version 0.2")
        if explicit and raw.get("positions_m") is None:
            raise ValueError("Schema 0.2 requires positions_m")
        result = cls(**{key: raw[key] for key in required}, positions_m=raw.get("positions_m"))
        expected_geometry = {
            "kind": "uniform_linear",
            "axis": "x",
            "origin": "array_center",
            "channel_order": f"MIC0_to_MIC{result.microphone_count - 1}_increasing_x",
            "forward_axis": "+y",
            "positive_angle_toward": "+x",
            "angle_unit": "deg",
        }
        if explicit:
            expected_geometry.update(
                kind="explicit_coordinates",
                axis="xyz",
                channel_order=f"MIC0_to_MIC{result.microphone_count - 1}_row_order",
            )
        geometry = raw.get("geometry", {})
        if not isinstance(geometry, Mapping):
            raise ValueError("geometry must be a mapping")
        if explicit and "coordinate_formula" in geometry:
            raise ValueError("Explicit coordinates must not carry a ULA coordinate_formula")
        for key, expected in expected_geometry.items():
            if geometry.get(key) != expected:
                raise ValueError(f"Unsupported geometry.{key}: {geometry.get(key)!r}")
        aperture = finite_real(raw.get("aperture_m"), "aperture_m")
        if explicit:
            positions = np.asarray(result.positions_m)
            expected_aperture = float(
                np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2))
            )
        else:
            expected_aperture = (result.microphone_count - 1) * result.spacing_m
        if not np.isclose(aperture, expected_aperture, rtol=0, atol=1e-10):
            raise ValueError("aperture_m disagrees with microphone count and spacing")
        clock = positive_int(raw.get("pdm_clock_hz"), "pdm_clock_hz")
        decimation = positive_int(raw.get("decimation_factor"), "decimation_factor")
        if clock != result.sample_rate_hz * decimation:
            raise ValueError("PDM clock / decimation must equal PCM sample rate")
        band = raw.get("spatial_processing_band_hz", [])
        if not isinstance(band, list) or len(band) != 2:
            raise ValueError("spatial_processing_band_hz must contain two values")
        low, high = [finite_real(v, "band edge") for v in band]
        if not 0 < low < high < result.sample_rate_hz / 2:
            raise ValueError("Processing band must be inside (0, Nyquist)")
        return result


@dataclass(frozen=True)
class STFTConfig:
    fft_size: int = 512
    hop_size: int = 256

    def __post_init__(self) -> None:
        positive_int(self.fft_size, "fft_size")
        positive_int(self.hop_size, "hop_size")
        if self.fft_size < 4 or self.fft_size % 2:
            raise ValueError("fft_size must be even and >= 4")
        if self.hop_size * 2 != self.fft_size:
            raise ValueError("M1 supports exactly 50% overlap; other hops are not implemented")

    @property
    def left_padding(self) -> int:
        return self.fft_size - self.hop_size
