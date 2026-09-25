"""Export bounded floating-point stage references; never a bit-accurate IP model."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from acoustic_array.beamforming.das import apply_weights, das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.dsp.stft import SpectralFrame, StreamingISTFT, StreamingSTFT

ROOT = Path(__file__).resolve().parents[1]


def export(destination: Path) -> None:
    """Create a new directory with deterministic inputs and verified stage outputs."""
    destination.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / "configs/array/w3_staggered.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    array = ArrayConfig.from_mapping(raw)
    cfg = STFTConfig(512, 256)
    n, h, length = cfg.fft_size, cfg.hop_size, 2121
    channels = array.microphone_count
    sample = np.arange(length)[:, None]
    mic = np.arange(channels)[None, :]
    # Channel-distinct electrical stimulus, not a simulated acoustic scene.
    stimulus = 0.08 * np.sin(2 * np.pi * sample * (317 + 61 * mic) / array.sample_rate_hz)
    stimulus += 0.01 * (-1.0) ** sample * (mic + 1) / channels
    stimulus += 0.001 * (mic - 7.5)
    stimulus[31 + 23 * np.arange(channels), np.arange(channels)] += 0.2
    pcm_integer = np.rint(stimulus * 2**23).astype("<i4")
    pcm = pcm_integer.astype(float) / 2**23
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    analyzer = StreamingSTFT(cfg, channels)
    frames = []
    for start in range(0, length, 137):
        frames.extend(analyzer.push(pcm[start : start + 137], sample_start=start))
    frames.extend(analyzer.flush())
    starts = np.array([frame.sample_start for frame in frames], dtype="<i8")
    padded = np.pad(pcm, ((h, n), (0, 0)))
    blocks = np.stack([padded[start + h : start + h + n] for start in starts])
    windowed = blocks * window[None, :, None]
    spectra = np.stack([frame.spectrum for frame in frames])
    np.testing.assert_allclose(spectra, np.fft.rfft(windowed, axis=1), atol=1e-12)
    files = {}

    def save(name: str, value: np.ndarray, axes: list[str]) -> None:
        dtype = "<c16" if np.iscomplexobj(value) else "<f8"
        if np.issubdtype(value.dtype, np.integer):
            dtype = "<i4" if value.dtype.itemsize == 4 else "<i8"
        data = np.asarray(value, dtype=dtype, order="C")
        path = destination / (name + ".bin")
        data.tofile(path)
        files[path.name] = {
            "shape": list(data.shape), "axes": axes, "dtype": dtype,
            "order": "C", "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    save("input_pcm_s24_in_s32", pcm_integer, ["sample", "microphone"])
    save("input_pcm", pcm, ["sample", "microphone"])
    save("frame_starts", starts, ["frame"])
    save("window", window, ["sample_in_frame"])
    save("windowed", windowed, ["frame", "sample_in_frame", "microphone"])
    save("input_spectrum", spectra, ["frame", "frequency", "microphone"])
    identity = np.zeros((n // 2 + 1, channels, 1), dtype=complex)
    identity[:, array.reference_microphone_index, 0] = 1
    angles = np.array([-45.0, 0.0, 45.0])
    weights = das_weights(array, cfg, angles)
    # Independent phase/sign and channel mapping check, including Nyquist policy.
    positions = np.asarray(array.positions_m)
    units = np.array([np.sin(np.deg2rad(angles)), np.cos(np.deg2rad(angles)), angles * 0])
    delay = -(positions - positions[array.reference_microphone_index]) @ units
    delay /= array.sound_speed_m_per_s
    frequencies = np.fft.rfftfreq(n, 1 / array.sample_rate_hz)
    expected_w = np.exp(-2j * np.pi * frequencies[:, None, None] * delay) / channels
    expected_w[[0, -1]] = expected_w[[0, -1]].real
    np.testing.assert_allclose(weights, expected_w, atol=1e-14, rtol=1e-13)
    errors = {}
    for case, w in (("identity_mic7", identity), ("das_three", weights)):
        y = np.stack([apply_weights(frame.spectrum, w) for frame in frames])
        direct = np.sum(spectra[:, :, :, None] * w.conj()[None], axis=2)
        np.testing.assert_allclose(y, direct, atol=1e-12, rtol=1e-12)
        inverse = np.fft.irfft(y, n=n, axis=1)
        synthesis = inverse * window[None, :, None]
        total = np.zeros((len(padded), w.shape[-1]))
        denominator = np.zeros(len(padded))
        for start, block in zip(starts, synthesis, strict=True):
            total[start + h : start + h + n] += block
            denominator[start + h : start + h + n] += window**2
        offline = total[h : h + length] / denominator[h : h + length, None]
        reconstructor = StreamingISTFT(cfg, w.shape[-1])
        chunks = [
            reconstructor.push(SpectralFrame(f.index, f.sample_start, out), valid_stop=length)
            for f, out in zip(frames, y, strict=True)
        ]
        reconstructor.finish(length)
        output = np.concatenate(chunks)
        np.testing.assert_allclose(output, offline, atol=1e-12, rtol=1e-12)
        if case == "identity_mic7":
            np.testing.assert_allclose(output[:, 0], pcm[:, 7], atol=1e-12, rtol=0)
        errors[case] = float(np.max(np.abs(output - offline)))
        save(case + "_weights", w, ["frequency", "microphone", "output"])
        save(case + "_spectrum", y, ["frame", "frequency", "output"])
        save(case + "_inverse", inverse, ["frame", "sample_in_frame", "output"])
        save(case + "_synthesis", synthesis, ["frame", "sample_in_frame", "output"])
        save(case + "_output", output, ["sample", "output"])
        save(case + "_chunk_lengths", np.array([len(c) for c in chunks]), ["frame"])
    save("ola_denominator", denominator[h : h + length], ["sample"])
    source_paths = [
        Path(__file__).resolve(), config_path,
        ROOT / "src/acoustic_array/dsp/stft.py",
        ROOT / "src/acoustic_array/beamforming/das.py",
        ROOT / "src/acoustic_array/core/config.py",
        ROOT / "src/acoustic_array/core/geometry.py",
    ]
    manifest = {
        "status": "floating_stage_reference_not_bit_accurate_not_acoustic_evidence",
        "sample_rate_hz": array.sample_rate_hz, "samples": length,
        "fft_size": n, "hop_size": h, "array_config": raw,
        "das_angles_deg": angles.tolist(), "seed": None,
        "input_scale": "signed 24-bit integer in low bits of int32; float = integer / 2**23",
        "complex_layout": "little-endian float64 real,imag pairs; no conjugation in file",
        "fft_scaling": "forward unscaled; inverse includes 1/N",
        "comparison": "float64 stage check: atol=1e-12 rtol=1e-12; NOT a hardware tolerance",
        "stream_vs_offline_max_abs": errors, "files": files,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(files), "checks": errors, "samples": length}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New output directory; never overwritten")
    export(parser.parse_args().destination)
