"""Development-only per-source DOA and static channel mismatch sweep."""

import argparse
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from acoustic_array.beamforming.constrained import constrained_weights
from acoustic_array.beamforming.das import das_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.core.geometry import directions
from acoustic_array.evaluation.components import component_ratios
from acoustic_array.evaluation.metrics import bandpass_for_evaluation, comparison_metrics
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.mismatch import apply_channel_mismatch


def _load(config_path: Path) -> tuple:
    config = load_json(config_path)
    if (
        set(config) != {"schema_version", "parent_run", "fft_sizes", "cases"}
        or config["schema_version"] != "0.1"
    ):
        raise ValueError("Unsupported mismatch config")
    parent = (config_path.parent / config["parent_run"]).resolve()
    manifest = load_json(parent / "manifest.json")
    for name in ["input.npz", "config.expanded.json"]:
        if sha256_file(parent / name) != manifest["outputs_sha256"][name]:
            raise ValueError("Parent artifact hash mismatch")
    expanded = load_json(parent / "config.expanded.json")
    array = ArrayConfig.from_mapping(expanded["array"])
    with np.load(parent / "input.npz", allow_pickle=False) as data:
        components, noise, mixture = data["components"], data["noise"], data["mixture"]
    angles = np.asarray(expanded["experiment"]["source_angles_deg"])
    directions(angles)
    if (
        components.ndim != 3
        or not components.size
        or components.shape[0] != len(angles)
        or components.shape[-1] != array.microphone_count
        or noise.shape != components.shape[1:]
        or mixture.shape != noise.shape
    ):
        raise ValueError("Invalid parent component shapes")
    if not all(np.isfinite(x).all() for x in [components, noise, mixture]):
        raise ValueError("Nonfinite parent samples")
    if not np.allclose(components.sum(axis=0) + noise, mixture, rtol=1e-12, atol=1e-14):
        raise ValueError("Parent components do not reconstruct mixture")
    sizes = config["fft_sizes"]
    if not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("fft_sizes must be nonempty and unique")
    stfts = [STFTConfig(n, n // 2) for n in sizes]
    cases = config["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be nonempty")
    names = set()
    trim = round(expanded["experiment"]["evaluation_trim_s"] * array.sample_rate_hz)
    if trim < max(sizes) or 2 * trim >= len(noise):
        raise ValueError("Insufficient evaluation trim")
    references = components[:, :, array.reference_microphone_index].T
    band_references = bandpass_for_evaluation(
        references, array.sample_rate_hz, tuple(expanded["experiment"]["evaluation_band_hz"])
    )
    if np.any(np.ptp(references[trim:-trim], axis=0) == 0) or np.any(
        np.ptp(band_references[trim:-trim], axis=0) == 0
    ):
        raise ValueError("Mismatch development sweep requires non-silent source references")
    for case in cases:
        if set(case) != {"name", "direction_errors_deg", "gain_db", "delay_samples"}:
            raise ValueError("Unsupported case fields")
        name = case["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Case names must be nonempty unique strings")
        names.add(name)
        errors = np.asarray(case["direction_errors_deg"], dtype=float)
        if errors.shape != angles.shape:
            raise ValueError("One direction error per source")
        directions(angles + errors)
        apply_channel_mismatch(
            np.zeros((1, array.microphone_count)), case["gain_db"], case["delay_samples"]
        )
        ref = array.reference_microphone_index
        if case["gain_db"][ref] != 0 or case["delay_samples"][ref] != 0:
            raise ValueError("Reference microphone must remain the calibration anchor")
    return config, parent, expanded, array, components, noise, stfts


def _evaluate_case(
    case: dict,
    stft: STFTConfig,
    array: ArrayConfig,
    base: dict,
    components: np.ndarray,
    noise: np.ndarray,
) -> tuple:
    angles = np.asarray(base["source_angles_deg"]) + np.asarray(case["direction_errors_deg"])
    constrained = constrained_weights(
        array,
        stft,
        angles,
        band_hz=tuple(base["constraint_band_hz"]),
        max_condition=base["max_condition"],
        min_white_noise_gain_db=base["min_white_noise_gain_db"],
    )
    trim = round(base["evaluation_trim_s"] * array.sample_rate_hz)
    crop = slice(trim, -trim)
    mixture = components.sum(axis=0) + noise
    ref = array.reference_microphone_index
    references = components[:, :, ref].T
    band = tuple(base["evaluation_band_hz"])
    ref_band = bandpass_for_evaluation(references, array.sample_rate_hz, band)
    mix_band = bandpass_for_evaluation(mixture[:, ref], array.sample_rate_hz, band)

    def replay(pcm: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return beamform_with_weights(pcm, array, stft, weights, block_size=base["block_size"])

    rows = []
    outputs = {}
    for method, weights in [
        ("das", das_weights(array, stft, angles)),
        ("fixed_constraint", constrained.weights),
    ]:
        output = replay(mixture, weights)
        outputs[method] = output
        filtered = bandpass_for_evaluation(output, array.sample_rate_hz, band)
        for k in range(len(components)):
            rows.append(
                {
                    "case": case["name"],
                    "fft_size": stft.fft_size,
                    "method": method,
                    "target": k,
                    "fullband": comparison_metrics(
                        references[crop, k], mixture[crop, ref], output[crop, k]
                    ),
                    "evaluation_band": comparison_metrics(
                        ref_band[crop, k], mix_band[crop], filtered[crop, k]
                    ),
                    "peak_abs": float(np.max(np.abs(output[:, k]))),
                }
            )
    processed = np.stack([replay(source, constrained.weights) for source in components])
    pn = replay(noise, constrained.weights)
    error = float(
        np.linalg.norm(processed.sum(axis=0) + pn - outputs["fixed_constraint"])
        / max(np.linalg.norm(outputs["fixed_constraint"]), 1e-30)
    )
    if not np.isfinite(error) or error > 1e-10:
        raise ValueError("Component replay failed")
    for row in rows:
        if row["method"] != "fixed_constraint":
            continue
        k = row["target"]
        others = [j for j in range(len(components)) if j != k]
        row["physical_fullband"] = component_ratios(
            processed[k, crop, k], processed[others, :, k].sum(axis=0)[crop], pn[crop, k]
        )
        fidelity = comparison_metrics(
            references[crop, k], processed[k, crop, k], processed[k, crop, k]
        )
        row["target_only_si_sdr_db"] = fidelity["output_si_sdr_db"]
        row["target_only_status"] = fidelity["metric_status"]["output_si_sdr_db"]
    diagnostics = {
        "case": case["name"],
        "fft_size": stft.fft_size,
        "reconstruction_relative_l2": error,
        "fallback_bin_outputs": int(np.char.startswith(constrained.status, "fallback").sum()),
    }
    return rows, outputs, constrained, diagnostics


def run_mismatch_experiment(config_path: Path, output_root: Path) -> Path:
    """Reuse a saved development scene; no automatic DOA or adaptive calibration."""
    config_path = config_path.resolve()
    config, parent, expanded, array, original, original_noise, stfts = _load(config_path)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run = output_root.resolve() / f"mismatch_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.expanded.json", {"sweep": config, "parent": expanded})
    rows = []
    diagnostics = []
    for i, case in enumerate(config["cases"]):
        components = apply_channel_mismatch(original, case["gain_db"], case["delay_samples"])
        noise = apply_channel_mismatch(original_noise, case["gain_db"], case["delay_samples"])
        for stft in stfts:
            scores, outputs, weights, diag = _evaluate_case(
                case, stft, array, expanded["experiment"], components, noise
            )
            rows.extend(scores)
            diagnostics.append(diag)
            stem = f"case_{i}_fft_{stft.fft_size}"
            np.savez_compressed(run / f"{stem}_outputs.npz", **outputs)
            np.savez_compressed(
                run / f"{stem}_weights.npz",
                weights=weights.weights,
                status=weights.status,
                wng_db=weights.white_noise_gain_db,
                residual=weights.constraint_residual,
            )
            print(f"Finished {case['name']}, N={stft.fft_size}", flush=True)
    write_json(
        run / "metrics.json",
        {
            "status": "development_only_not_acceptance",
            "rows": rows,
            "diagnostics": diagnostics,
            "activity_mask": "none_whole_cropped_clip",
            "reference": (
                f"MIC{array.reference_microphone_index} source component; "
                "reference gain/delay anchored at zero"
            ),
            "noise_transfer": "noise before mismatch; gain/delay also applied to noise",
        },
    )
    package = Path(__file__).resolve().parents[1]
    write_json(
        run / "manifest.json",
        {
            "created_utc": stamp,
            "parent_run": str(parent),
            "parent_manifest_sha256": sha256_file(parent / "manifest.json"),
            "config_sha256": canonical_hash(config),
            "config_file_sha256": sha256_file(config_path),
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "dependencies": {
                    n: importlib.metadata.version(n)
                    for n in ["numpy", "scipy", "acoustic-array-fpga"]
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
    print(run_mismatch_experiment(args.config, args.output_root))


if __name__ == "__main__":
    main()
