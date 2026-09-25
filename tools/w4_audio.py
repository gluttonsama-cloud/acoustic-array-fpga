"""Frozen W4 three-port audio comparison; run from the repository root."""

import argparse
import importlib.metadata
import itertools
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_array.beamforming.das import das_weights
from acoustic_array.beamforming.mpdr import mixture_covariance, mpdr_weights
from acoustic_array.core.config import ArrayConfig, STFTConfig
from acoustic_array.io.artifacts import canonical_hash, load_json, sha256_file, write_json
from acoustic_array.pipelines.room_experiment import _score
from acoustic_array.pipelines.streaming import beamform_with_weights
from acoustic_array.simulation.active_scene import normalize_active_scene
from acoustic_array.simulation.room import RoomRIR, propagate_room

try:
    from tools.w4_boundary import verify_run
except ModuleNotFoundError:
    from w4_boundary import verify_run


CASES = ("r2_gap15_rt0.4", "r2_gap45_rt0.4", "r3_gap15_rt0.4", "r3_gap45_rt0.4")
BRANCHES = ("baseline", "onset", "oracle")
METHODS = ("das", "mpdr")


def match_ports(truth_deg: np.ndarray, ports_deg: np.ndarray, tolerance_deg: float) -> list[dict]:
    """Global angular assignment; ties preserve most accurate ports, then index order."""
    truth = np.asarray(truth_deg, dtype=float)
    ports = np.asarray(ports_deg, dtype=float)
    if (
        truth.shape != (3,)
        or ports.shape != (3,)
        or not np.isfinite(truth).all()
        or not np.isfinite(ports).all()
    ):
        raise ValueError("Three finite truth and port angles required")
    if not np.isfinite(tolerance_deg) or tolerance_deg < 0:
        raise ValueError("Invalid angle tolerance")
    choices = []
    for permutation in itertools.permutations(range(3)):
        errors = np.abs(truth - ports[list(permutation)])
        choices.append((float(errors.sum()), -int((errors <= tolerance_deg).sum()), permutation))
    minimum = min(choice[0] for choice in choices)
    tied = [choice for choice in choices if abs(choice[0] - minimum) <= 1e-9]
    _, _, assignment = min(tied, key=lambda choice: (choice[1], choice[2]))
    return [
        {
            "target": k,
            "port": assignment[k],
            "error_deg": float(abs(truth[k] - ports[assignment[k]])),
            "localized_within_tolerance": bool(
                abs(truth[k] - ports[assignment[k]]) <= tolerance_deg
            ),
        }
        for k in range(3)
    ]


def select_angles(rows: list[dict], cases: tuple[str, ...]) -> dict[str, list[float]]:
    """Take only full, reverberant triple-source rows from a frozen run."""
    selected = {}
    for row in rows:
        if row.get("mode") != "reverberant" or row.get("source_subset") != [0, 1, 2]:
            continue
        case = row.get("case")
        if case not in cases or case in selected or row.get("method", "normmusic") != "normmusic":
            raise ValueError("Unexpected or duplicate W4 direction row")
        peaks = row.get("peaks")
        if not isinstance(peaks, list) or len(peaks) != 3:
            raise ValueError(f"Incomplete three-port estimate: {case}")
        angles = sorted(float(peak["angle_deg"]) for peak in peaks)
        if not np.isfinite(angles).all() or len(set(angles)) != 3:
            raise ValueError(f"Invalid W4 directions: {case}")
        selected[case] = angles
    if set(selected) != set(cases):
        raise ValueError("Incomplete four-case W4 directions")
    return selected


def decision_gate(rows: list[dict], cases: tuple[str, ...], cfg: dict) -> dict:
    """A missing, nonfinite, or duplicated target result is an explicit gate failure."""
    pairs = [(r.get("case"), r.get("branch"), r.get("method")) for r in rows]
    expected = {(c, b, m) for c in cases for b in BRANCHES for m in METHODS}
    complete = len(pairs) == len(expected) and set(pairs) == expected
    values = {}
    if complete:
        for row in rows:
            for reference in ("direct_reference", "reverberant_reference"):
                for band in ("fullband", "evaluation_band"):
                    reference_summary = row.get(reference)
                    if not isinstance(reference_summary, dict):
                        complete = False
                        break
                    summary = reference_summary.get(band)
                    if not isinstance(summary, dict):
                        complete = False
                        break
                    targets = summary.get("targets", [])
                    if not isinstance(targets, list) or len(targets) != 3:
                        complete = False
                        break
                    scores = []
                    for target in targets:
                        if not isinstance(target, dict) or target.get("status") != "ok":
                            complete = False
                            break
                        count = target.get("active_samples")
                        metrics = target.get("metrics")
                        if (
                            not isinstance(count, (int, float))
                            or not np.isfinite(count)
                            or count <= 0
                            or not isinstance(metrics, dict)
                            or any(
                                key not in metrics
                                or not isinstance(metrics[key], (int, float))
                                or not np.isfinite(metrics[key])
                                for key in ("input_si_sdr_db", "output_si_sdr_db", "si_sdri_db")
                            )
                        ):
                            complete = False
                            break
                        scores.append(metrics["si_sdri_db"])
                    if not complete:
                        break
                    if reference == "direct_reference" and band == "fullband":
                        values[(row["case"], row["branch"], row["method"])] = min(scores)
                if not complete:
                    break
            if not complete:
                break
    if not complete:
        return {"passed": False, "status": "incomplete_or_nonfinite", "case_minima_db": {}}
    comparisons = {
        case: {branch: float(values[(case, branch, "mpdr")]) for branch in BRANCHES}
        for case in cases
    }
    differences = [comparisons[case]["onset"] - comparisons[case]["baseline"] for case in cases]
    median_gain = float(
        np.median([comparisons[case]["onset"] for case in cases])
        - np.median([comparisons[case]["baseline"] for case in cases])
    )
    passed = (
        median_gain >= cfg["minimum_median_gain_db"]
        and min(differences) >= -cfg["maximum_case_regression_db"]
    )
    return {
        "passed": bool(passed),
        "status": "passed" if passed else "threshold_not_met",
        "case_minima_db": comparisons,
        "onset_minus_baseline_by_case_db": dict(zip(cases, map(float, differences), strict=True)),
        "median_of_case_minima_gain_db": median_gain,
    }


def design_weights(
    mixture: np.ndarray,
    array: ArrayConfig,
    stft: STFTConfig,
    branches: dict[str, list[float]],
    cfg: dict,
) -> tuple[dict, np.ndarray, int]:
    """Weight design only sees microphone mixture and three proposed angle lists."""
    calibration = round(cfg["calibration_seconds"] * array.sample_rate_hz)
    covariance, frames = mixture_covariance(mixture[:calibration], array, stft)
    designs = {}
    for branch, angles in branches.items():
        das = das_weights(array, stft, angles)
        mpdr = mpdr_weights(
            covariance,
            array,
            stft,
            angles,
            diagonal_loading=cfg["diagonal_loading"],
            band_hz=tuple(cfg["band_hz"]),
            max_condition=cfg["max_condition"],
            min_white_noise_gain_db=cfg["min_wng_db"],
        )
        designs[(branch, "das")] = {
            "weights": das,
            "status": np.full((stft.fft_size // 2 + 1, 3), "das", dtype="U32"),
        }
        designs[(branch, "mpdr")] = {
            "weights": mpdr.weights,
            "status": mpdr.status,
            "condition": mpdr.condition_number,
            "residual": mpdr.constraint_residual,
            "wng_db": mpdr.white_noise_gain_db,
        }
    return designs, covariance, frames


def _inputs(cfg: dict, config_path: Path) -> tuple[dict, dict, dict]:
    required = {
        "schema_version",
        "parent_run",
        "baseline_run",
        "onset_run",
        "cases",
        "fft_size",
        "calibration_seconds",
        "band_hz",
        "evaluation_band_hz",
        "diagonal_loading",
        "min_wng_db",
        "max_condition",
        "matching_tolerance_deg",
        "minimum_median_gain_db",
        "maximum_case_regression_db",
    }
    if set(cfg) != required or cfg["schema_version"] != "0.1" or tuple(cfg["cases"]) != CASES:
        raise ValueError("W4 audio config does not match the preregistered cases")
    fixed = {
        "fft_size": 512,
        "calibration_seconds": 3.0,
        "band_hz": [100, 5000],
        "evaluation_band_hz": [500, 5000],
        "diagonal_loading": 0.1,
        "min_wng_db": -10,
        "max_condition": 1e6,
        "matching_tolerance_deg": 5,
        "minimum_median_gain_db": 1.0,
        "maximum_case_regression_db": 1.0,
    }
    if any(cfg[k] != v for k, v in fixed.items()):
        raise ValueError("W4 audio settings changed from registration")
    names = {
        k: (config_path.parent / cfg[f"{k}_run"]).resolve() for k in ("parent", "baseline", "onset")
    }
    if (
        names["parent"].name != "g1_room_20260923T171109_297143Z"
        or names["baseline"].name != "w4_boundary_20260924T150817_390183Z"
        or names["onset"].name != "w4_onset_20260925T034430_129538Z"
    ):
        raise ValueError("W4 audio parent run changed")
    evidence = {
        name: verify_run(path, require_completion=name != "parent") for name, path in names.items()
    }
    baseline_cfg = load_json(names["baseline"] / "config.expanded.json")
    onset_cfg = load_json(names["onset"] / "config.expanded.json")
    if (
        baseline_cfg["parent"]["manifest_sha256"] != evidence["parent"]["manifest_sha256"]
        or onset_cfg["parent"]["manifest_sha256"] != evidence["parent"]["manifest_sha256"]
        or onset_cfg["baseline"]["manifest_sha256"] != evidence["baseline"]["manifest_sha256"]
    ):
        raise ValueError("W4 run parent hash chain differs")
    for prior in (baseline_cfg, onset_cfg):
        if (
            prior["config"]["observation_seconds"] != cfg["calibration_seconds"]
            or prior["config"]["fft_size"] != cfg["fft_size"]
            or prior["config"]["cases"] != cfg["cases"]
        ):
            raise ValueError("Frozen localization settings differ")
    angles = {
        "baseline": select_angles(load_json(names["baseline"] / "metrics.json")["rows"], CASES),
        "onset": select_angles(load_json(names["onset"] / "metrics.json")["rows"], CASES),
    }
    return names, evidence, angles


def run_audio(config_path: Path, output_root: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    config_path = config_path.resolve()
    cfg = load_json(config_path)
    names, evidence, angles = _inputs(cfg, config_path)
    parent = names["parent"]
    expanded = load_json(parent / "config.expanded.json")
    original = expanded["config"]
    array = ArrayConfig.from_mapping(expanded["array"])
    stft = STFTConfig(cfg["fft_size"], cfg["fft_size"] // 2)
    calibration = round(cfg["calibration_seconds"] * array.sample_rate_hz)
    trim = original["trim_samples"]
    with np.load(parent / "sources.npz") as archive:
        sources = archive["sources"].copy()
    if (
        sources.shape[0] != 3
        or calibration + trim >= sources.shape[1] - trim
        or trim != round(0.65 * array.sample_rate_hz)
    ):
        raise ValueError("W4 evaluation interval is not 3.65 to 9.35 seconds")
    diagnoses = {
        d["case"]: d
        for d in json.loads((parent / "room_diagnostics.json").read_text(encoding="utf-8"))
    }
    if not set(CASES) <= diagnoses.keys() or array.reference_microphone_index != 7:
        raise ValueError("W4 scene grid or reference channel differs")
    run = output_root.resolve() / f"w4_audio_{datetime.now(UTC):%Y%m%dT%H%M%S_%fZ}"
    run.mkdir(parents=True, exist_ok=False)
    write_json(
        run / "config.expanded.json",
        {
            "config": cfg,
            "array": expanded["array"],
            "input_runs": evidence,
            "config_sha256": sha256_file(config_path),
            "calibration_samples": [0, calibration],
            "evaluation_samples": [calibration + trim, sources.shape[1] - trim],
            "policy": (
                "estimated ports sorted by angle; oracle angle only; "
                "far-field steering; mixture-only covariance"
            ),
        },
    )
    for source in [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__).resolve(),
        root / "tools/w4_boundary.py",
    ]:
        target = run / "source_snapshot" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    rows = []
    for case in CASES:
        diagnosis = diagnoses[case]
        with np.load(parent / f"{case}_reconstruction.npz") as archive:
            rir = RoomRIR(
                archive["rirs"],
                archive["direct_rirs"],
                archive["sources_m"],
                archive["microphones_m"],
                archive["direct_distances_m"],
                diagnosis["target_rt60_s"],
                diagnosis["energy_absorption"],
            )
            base, direct = propagate_room(sources, array, rir)
            built = normalize_active_scene(
                base,
                array,
                activity_components=direct,
                active_rms=original["active_rms"],
                noise_snr_db=original["noise_snr_db"],
                seed=original["seed"],
                trim_samples=trim,
            )
            if not np.array_equal(built.source_gains, archive["gains"]) or not np.array_equal(
                built.masks, archive["masks"]
            ):
                raise ValueError("Frozen parent scene normalization differs")
        offset = rir.source_positions_m - np.asarray(original["room"]["array_center_m"])
        truth = np.rad2deg(np.arctan2(offset[:, 0], offset[:, 1]))
        branches = {name: angles[name][case] for name in ("baseline", "onset")}
        branches["oracle"] = sorted(map(float, truth))
        mixture = built.scene.mixture
        designs, covariance, frames = design_weights(mixture, array, stft, branches, cfg)
        np.savez_compressed(run / f"{case}_calibration.npz", covariance=covariance, frames=frames)
        outputs = {}
        for (branch, method), design in designs.items():
            tail = beamform_with_weights(
                mixture[calibration:],
                array,
                stft,
                design["weights"],
                block_size=original["block_size"],
            )
            output = np.pad(tail, ((calibration, 0), (0, 0)))
            if output.shape != (len(mixture), 3) or not np.isfinite(output).all():
                raise ValueError("Incomplete or nonfinite W4 output")
            outputs[(branch, method)] = output
            np.savez_compressed(run / f"{case}_{branch}_{method}.npz", output=output, **design)
        # One listening scale for the complete case, including the microphone reference.
        listen_peak = max(
            float(np.max(np.abs(mixture[:, array.reference_microphone_index]))),
            *(float(np.max(np.abs(x))) for x in outputs.values()),
        )
        gain = 0.95 / listen_peak if listen_peak > 0 else 1.0
        sf.write(
            run / f"{case}_mixture.wav",
            mixture[:, array.reference_microphone_index] * gain,
            array.sample_rate_hz,
            subtype="PCM_24",
        )
        direct_scaled = direct * built.source_gains[:, None, None]
        references = direct_scaled[:, :, array.reference_microphone_index].T
        reverberant = built.scene.components[:, :, array.reference_microphone_index].T
        masks = built.masks.copy()
        masks[:, : calibration + trim] = False
        masks[:, sources.shape[1] - trim :] = False
        for (branch, method), output in outputs.items():
            sf.write(
                run / f"{case}_{branch}_{method}.wav",
                output * gain,
                array.sample_rate_hz,
                subtype="PCM_24",
            )
            for port in range(3):
                sf.write(
                    run / f"{case}_{branch}_{method}_port{port}.wav",
                    output[:, port] * gain,
                    array.sample_rate_hz,
                    subtype="PCM_24",
                )
            assignment = match_ports(
                truth, np.asarray(branches[branch]), cfg["matching_tolerance_deg"]
            )
            ordered = output[:, [item["port"] for item in assignment]]
            rows.append(
                {
                    "case": case,
                    "branch": branch,
                    "method": method,
                    "truth_deg_scoring_only": truth.tolist(),
                    "port_angles_deg": branches[branch],
                    "assignment_scoring_only": assignment,
                    "localized_targets": sum(
                        item["localized_within_tolerance"] for item in assignment
                    ),
                    "localization_failures": sum(
                        not item["localized_within_tolerance"] for item in assignment
                    ),
                    "calibration_frames": frames,
                    "active_samples_by_target": [int(m.sum()) for m in masks],
                    "listening_gain": gain,
                    "direct_reference": _score(
                        references,
                        mixture[:, array.reference_microphone_index],
                        ordered,
                        masks,
                        array,
                        trim,
                    ),
                    "reverberant_reference": _score(
                        reverberant,
                        mixture[:, array.reference_microphone_index],
                        ordered,
                        masks,
                        array,
                        trim,
                    ),
                }
            )
        write_json(
            run / "metrics.json",
            {
                "status": "in_progress",
                "completed_cases": len(rows) // 6,
                "expected_cases": len(CASES),
                "rows": rows,
            },
        )
        print(case, "complete", flush=True)
    gate = decision_gate(rows, CASES, cfg)
    write_json(
        run / "metrics.json",
        {
            "status": "development_complete",
            "completed_cases": len(CASES),
            "expected_cases": len(CASES),
            "reference_channel_index": array.reference_microphone_index,
            "evaluation_band_hz": cfg["evaluation_band_hz"],
            "matching_policy": (
                "global minimum total absolute angular error; "
                "tie most <=5deg then lexicographic ports"
            ),
            "wav_ports": "port0 through port2 are zero-based angle-sorted inference ports; "
            "see each row's port_angles_deg; all WAVs in a case share one gain",
            "rows": rows,
            "decision_gate": gate,
        },
    )
    manifest = {
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "input_config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_hash(cfg),
        "parent_manifest_sha256": evidence["parent"]["manifest_sha256"],
        "baseline_manifest_sha256": evidence["baseline"]["manifest_sha256"],
        "onset_manifest_sha256": evidence["onset"]["manifest_sha256"],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "soundfile", "rir-generator", "acoustic-array-fpga")
        },
        "outputs_sha256": {
            p.relative_to(run).as_posix(): sha256_file(p)
            for p in sorted(run.rglob("*"))
            if p.is_file()
        },
    }
    write_json(run / "manifest.json", manifest)
    pending = run / "completion.pending.json"
    write_json(
        pending,
        {
            "run_status": "complete",
            "completed_cases": len(CASES),
            "manifest_sha256": sha256_file(run / "manifest.json"),
        },
    )
    pending.replace(run / "completion.json")
    print(run, gate, flush=True)
    return run


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/w4_audio.json")
    parser.add_argument("--output-root", type=Path, default=root / "artifacts/runs")
    args = parser.parse_args()
    run_audio(args.config, args.output_root)


if __name__ == "__main__":
    main()
