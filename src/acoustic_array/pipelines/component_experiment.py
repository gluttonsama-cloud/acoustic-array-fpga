"""Audit a saved nominal fixed-weight speech run by linear component replay."""

import argparse
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.evaluation.components import component_ratios, spectral_power_partition
from acoustic_array.evaluation.metrics import bandpass_for_evaluation, comparison_metrics
from acoustic_array.io.artifacts import load_json, sha256_file, write_json
from acoustic_array.pipelines.streaming import beamform_with_weights


def _target_diagnostics(
    before: list[np.ndarray],
    after: list[np.ndarray],
    sample_rate_hz: int,
    trim: int,
    evaluation_band: tuple,
) -> dict:
    """Evaluate one target using identical crops and physical component definitions."""
    crop = slice(trim, -trim)
    groups = {}
    for label, band in [
        ("fullband", None),
        ("evaluation_band", evaluation_band),
    ]:

        def filtered(x: np.ndarray, band: tuple | None = band) -> np.ndarray:
            return (x if band is None else bandpass_for_evaluation(x, sample_rate_hz, band))[crop]

        inp = component_ratios(*[filtered(x) for x in before])
        out = component_ratios(*[filtered(x) for x in after])
        gains = {}
        for key in ["sir", "snr"]:
            a, b = inp[key]["db"], out[key]["db"]
            gains[key + "_improvement_db"] = None if a is None or b is None else b - a
        groups[label] = {
            "input": inp,
            "output": out,
            **gains,
        }
        reference = filtered(before[0])
        if np.ptp(reference) == 0:
            groups[label]["target_only_si_sdr"] = {
                "db": None,
                "status": "undefined_silent_reference",
            }
        else:
            fidelity = comparison_metrics(reference, filtered(after[0]), filtered(after[0]))
            groups[label]["target_only_si_sdr"] = {
                "db": fidelity["output_si_sdr_db"],
                "status": fidelity["metric_status"]["output_si_sdr_db"],
            }
    partitions = {
        name: spectral_power_partition(x[crop], sample_rate_hz)
        for name, x in zip(["target", "interference", "noise"], after, strict=True)
    }
    return {"metrics": groups, "output_spectral_powers": partitions}


def _load_parent_artifacts(parent_run: Path) -> tuple:
    """Load only checksum-verified parent arrays and enforce replay dimensions."""
    parent_run = parent_run.resolve()
    manifest = load_json(parent_run / "manifest.json")

    def checked(name: str) -> Path:
        path = parent_run / name
        if sha256_file(path) != manifest["outputs_sha256"][name]:
            raise ValueError(f"Parent artifact hash mismatch: {name}")
        return path

    expanded = load_json(checked("config.expanded.json"))
    config = expanded["experiment"]
    if len(config["fft_sizes"]) != 1:
        raise ValueError("Component audit requires a single FFT size")
    size = config["fft_sizes"][0]
    nominal_index = config["direction_offsets_deg"].index(0)
    weight_name = f"weights_{size}_offset_index_{nominal_index}.npz"
    with np.load(checked(weight_name), allow_pickle=False) as saved:
        weights = saved["weights"]
    with np.load(checked("input.npz"), allow_pickle=False) as saved:
        components, noise, mixture = saved["components"], saved["noise"], saved["mixture"]
    nominal = np.load(checked(f"fixed_constraint_{size}.npy"), allow_pickle=False)
    array = ArrayConfig.from_mapping(expanded["array"])
    stft = STFTConfig(size, size // 2)
    if (
        components.ndim != 3
        or noise.ndim != 2
        or mixture.shape != noise.shape
        or components.shape[1:] != noise.shape
        or not len(components)
        or weights.ndim != 3
        or weights.shape[2] != len(components)
        or nominal.shape != (len(noise), len(components))
        or not all(np.isfinite(x).all() for x in [components, noise, mixture, nominal])
    ):
        raise ValueError("Invalid saved component shapes or nonfinite samples")

    return expanded, weights, components, noise, mixture, nominal, array, stft


def run_component_experiment(parent_run: Path, output_root: Path) -> Path:
    """Validate saved artifacts, replay unchanged weights, and archive a component audit."""
    parent_run = parent_run.resolve()
    expanded, weights, components, noise, mixture, nominal, array, stft = _load_parent_artifacts(
        parent_run
    )
    config = expanded["experiment"]

    def replay(pcm: np.ndarray) -> np.ndarray:
        return beamform_with_weights(pcm, array, stft, weights, block_size=config["block_size"])

    # The saved parent-run weights are never re-estimated for individual sources.
    processed = np.stack([replay(source) for source in components])
    processed_noise = replay(noise)
    recombined = processed.sum(axis=0) + processed_noise
    relative_error = float(
        np.linalg.norm(recombined - nominal) / max(np.linalg.norm(nominal), 1e-30)
    )
    if relative_error > 1e-10:
        raise ValueError("Component replay does not reconstruct parent output")
    if not np.allclose(components.sum(axis=0) + noise, mixture, rtol=1e-12, atol=1e-14):
        raise ValueError("Parent input components do not reconstruct mixture")
    trim = round(config["evaluation_trim_s"] * array.sample_rate_hz)
    if trim <= 0 or 2 * trim >= len(noise):
        raise ValueError("Invalid evaluation trim")
    rows = []
    for k in range(len(components)):
        others = [j for j in range(len(components)) if j != k]
        before = [
            components[k, :, array.reference_microphone_index],
            components[others, :, array.reference_microphone_index].sum(axis=0),
            noise[:, array.reference_microphone_index],
        ]
        after = [processed[k, :, k], processed[others, :, k].sum(axis=0), processed_noise[:, k]]
        rows.append(
            {
                "target": k,
                **_target_diagnostics(
                    before, after, array.sample_rate_hz, trim, tuple(config["evaluation_band_hz"])
                ),
            }
        )
    return _write_audit(
        parent_run, output_root, expanded, processed, processed_noise, rows, relative_error, trim
    )


def _write_audit(
    parent_run: Path,
    output_root: Path,
    expanded: dict,
    processed: np.ndarray,
    processed_noise: np.ndarray,
    rows: list[dict],
    relative_error: float,
    trim: int,
) -> Path:
    """Archive replay components, metrics, runtime, and provenance."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"components_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        run / "components.npz", source_outputs=processed, noise_output=processed_noise
    )
    write_json(
        run / "metrics.json",
        {
            "status": "development_only_not_acceptance",
            "rows": rows,
            "reconstruction_relative_l2": relative_error,
            "trim_each_end": trim,
            "activity_mask": "none_whole_cropped_clip",
            "interference": "sum_waveforms_before_power",
            "spectral_partition": "rectangular_DFT_Parseval_no_demean_after_crop",
            "sir_definition": "physical_component_power_ratio_not_BSS_EVAL_projection",
        },
    )
    package_root = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "created_utc": stamp,
            "parent_run": str(parent_run),
            "parent_manifest_sha256": sha256_file(parent_run / "manifest.json"),
            "expanded_config": expanded,
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
                p.relative_to(package_root).as_posix(): sha256_file(p)
                for p in sorted(package_root.rglob("*.py"))
            },
            "outputs_sha256": {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()},
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = parser.parse_args()
    print(run_component_experiment(args.parent_run, args.output_root))


if __name__ == "__main__":
    main()
