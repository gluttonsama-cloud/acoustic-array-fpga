"""Active-RMS scene generation on explicitly development-only local speech."""

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
from acoustic_array.evaluation.metrics import bandpass_for_evaluation, comparison_metrics
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.io.speech import load_speech_sources
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import build_active_scene


def run_active_scene_experiment(config_path: Path, output_root: Path) -> Path:
    """Build one scene and compare fixed weights; not heldout acceptance."""
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    fields = {
        "schema_version",
        "array_config",
        "speech_manifest",
        "duration_s",
        "seed",
        "angles_deg",
        "active_rms",
        "noise_snr_db",
        "trim_samples",
        "fft_sizes",
        "block_size",
        "constraint_band_hz",
        "min_wng_db",
        "max_condition",
    }
    if set(cfg) != fields or cfg["schema_version"] != "0.1":
        raise ValueError("Invalid active scene config")
    array_path = (config_path.parent / cfg["array_config"]).resolve()
    speech_path = (config_path.parent / cfg["speech_manifest"]).resolve()
    array_raw = load_json(array_path)
    array = ArrayConfig.from_mapping(array_raw)
    positive_int(cfg["block_size"], "block_size")
    positive_int(cfg["trim_samples"], "trim_samples")
    duration = finite_real(cfg["duration_s"], "duration_s")
    if not 0.1 <= duration <= 30:
        raise ValueError("Development duration must be 0.1 to 30 seconds")
    if array.sample_rate_hz <= 10000:
        raise ValueError("Evaluation band requires sample rate above 10000 Hz")
    sizes = cfg["fft_sizes"]
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("Invalid FFT list")
    stfts = [STFTConfig(n, n // 2) for n in sizes]
    if cfg["trim_samples"] < max(sizes):
        raise ValueError("Trim must cover largest FFT")
    sources, records = load_speech_sources(
        speech_path, array.sample_rate_hz, cfg["duration_s"], 0.04
    )
    if len(sources) != 3:
        raise ValueError("S1 development requires three sources")
    # Loader's common whole-clip scaling is only preprocessing. Final level is set below.
    built = build_active_scene(
        sources,
        array,
        cfg["angles_deg"],
        active_rms=cfg["active_rms"],
        noise_snr_db=cfg["noise_snr_db"],
        seed=cfg["seed"],
        trim_samples=cfg["trim_samples"],
    )
    weight_sets = [
        constrained_weights(
            array,
            s,
            cfg["angles_deg"],
            band_hz=tuple(cfg["constraint_band_hz"]),
            max_condition=cfg["max_condition"],
            min_white_noise_gain_db=cfg["min_wng_db"],
        )
        for s in stfts
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"active_scene_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    scene = built.scene
    ref = array.reference_microphone_index
    trim = cfg["trim_samples"]
    crop = slice(trim, -trim)
    references = scene.components[:, :, ref].T
    mixture = scene.mixture[:, ref]
    rb = bandpass_for_evaluation(references, array.sample_rate_hz, (500, 5000))
    mb = bandpass_for_evaluation(mixture, array.sample_rate_hz, (500, 5000))
    np.savez_compressed(
        run / "reconstruction_input.npz",
        sources=sources,
        masks=built.masks,
        source_gains=built.source_gains,
    )
    rows = []
    for stft, constrained in zip(stfts, weight_sets, strict=True):
        np.savez_compressed(
            run / f"weights_{stft.fft_size}.npz",
            weights=constrained.weights,
            status=constrained.status,
            wng_db=constrained.white_noise_gain_db,
        )
        for method, weights in [
            ("das", das_weights(array, stft, cfg["angles_deg"])),
            ("fixed_constraint", constrained.weights),
        ]:
            out = beamform_with_weights(
                scene.mixture, array, stft, weights, block_size=cfg["block_size"]
            )
            ob = bandpass_for_evaluation(out, array.sample_rate_hz, (500, 5000))
            for k in range(len(sources)):
                row = {"fft_size": stft.fft_size, "method": method, "target": k}
                for label, r, m, e in [
                    ("fullband", references, mixture, out),
                    ("evaluation_band", rb, mb, ob),
                ]:
                    row[label] = {
                        "whole": comparison_metrics(r[crop, k], m[crop], e[crop, k]),
                        "active": masked_comparison(
                            r[crop, k],
                            m[crop],
                            e[crop, k],
                            built.masks[k, crop],
                            array.sample_rate_hz,
                        ),
                    }
                rows.append(row)
    expanded = {
        "config": cfg,
        "array": array_raw,
        "speech_sources": records,
        "activity": asdict(ActivityConfig()),
    }
    write_json(run / "config.expanded.json", expanded)
    write_json(
        run / "metrics.json",
        {
            "status": "development_only_not_B1",
            "rows": rows,
            "source_active_rms": built.active_rms.tolist(),
            "reference_noise_rms": built.reference_noise_rms,
            "measured_active_snr_db": built.measured_active_snr_db.tolist(),
            "active_fraction_after_trim": built.masks[:, crop].mean(axis=1).tolist(),
            "mask_policy": "frozen before final normalization; computed after source preprocessing",
            "noise_policy": "exact MIC reference full cropped RMS; per-target active SNR measured",
            "archive_policy": (
                "sources/masks/gains plus config seed; "
                "rebuild via build_active_scene; no full PCM archive"
            ),
        },
    )
    package = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "created_utc": stamp,
            "config_sha256": canonical_hash(expanded),
            "input_files_sha256": {
                str(p): sha256_file(p) for p in [config_path, array_path, speech_path]
            },
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in ["numpy", "scipy", "acoustic-array-fpga"]
                },
            },
            "argv": sys.argv,
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
    print(run_active_scene_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
