"""Failure-preserving scene statistics; never treats output channels as scenes."""

import numpy as np

from acoustic_array.core.config import positive_int


def _number(value: float) -> dict:
    return {
        "db": float(value) if np.isfinite(value) else None,
        "status": "finite" if np.isfinite(value) else "negative_infinity",
    }


def summarize_s1(records: list[dict], *, bootstrap_samples: int = 2000, seed: int = 0) -> dict:
    """Each record: unique scene_id, status ok/failed/silent, three finite target scores if ok."""
    positive_int(bootstrap_samples, "bootstrap_samples")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Invalid seed")
    ids = set()
    scores = []
    failed = silent = 0
    for r in records:
        name = r["scene_id"]
        status = r["status"]
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Duplicate/invalid scene_id")
        ids.add(name)
        if status == "silent":
            silent += 1
            continue
        if status == "failed":
            failed += 1
            scores.append(-np.inf)
            continue
        if status != "ok":
            raise ValueError("Invalid scene status")
        values = np.asarray(r["target_si_sdri_db"], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Successful S1 scene requires three finite target scores")
        scores.append(float(values.min()))
    if not scores:
        return {"status": "no_effect_scenes", "effect_scenes": 0, "silent_scenes": silent}
    values = np.asarray(scores)
    rng = np.random.default_rng(seed)

    def quant(x: list | np.ndarray, q: float) -> float:
        return float(np.quantile(x, q, method="inverted_cdf"))

    boot_median = []
    boot_p10 = []
    for _ in range(bootstrap_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_median.append(quant(sample, 0.5))
        boot_p10.append(quant(sample, 0.1))
    return {
        "status": "descriptive_only_not_acceptance",
        "effect_scenes": len(values),
        "silent_scenes": silent,
        "failed_scenes": failed,
        "failure_fraction": failed / len(values),
        "finite_no_improvement_scenes": int(np.sum(np.isfinite(values) & (values <= 0))),
        "positive_fraction": float(np.mean(values > 0)),
        "median": _number(quant(values, 0.5)),
        "p10": _number(quant(values, 0.1)),
        "median_ci95": [_number(quant(boot_median, q)) for q in [0.025, 0.975]],
        "p10_ci95": [_number(quant(boot_p10, q)) for q in [0.025, 0.975]],
        "quantile_method": "inverted_cdf",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "scene_scores": [_number(v) for v in scores],
    }


def summarize_s1_stratified(
    records: list[dict], strata: dict[str, int], *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict:
    """Fixed-composition scene bootstrap; unplanned silent inputs are failures upstream."""
    if not strata or any(not isinstance(k, str) or not k for k in strata):
        raise ValueError("Expected named strata")
    for count in strata.values():
        positive_int(count, "stratum count")
    if any(r.get("stratum") not in strata or r["status"] == "silent" for r in records):
        raise ValueError("Unknown stratum or unplanned silent scene")
    if any(sum(r["stratum"] == s for r in records) != n for s, n in strata.items()):
        raise ValueError("Stratum counts differ from design")
    # Reuse validation and point estimates; replace the ordinary bootstrap intervals below.
    result = summarize_s1(records, bootstrap_samples=bootstrap_samples, seed=seed)
    values = np.asarray(
        [r["db"] if r["status"] == "finite" else -np.inf for r in result["scene_scores"]]
    )
    groups = [values[[r["stratum"] == s for r in records]] for s in strata]
    rng = np.random.default_rng(seed)
    draws = np.concatenate(
        [rng.choice(group, size=(bootstrap_samples, len(group)), replace=True) for group in groups],
        axis=1,
    )
    for name, q in [("median", 0.5), ("p10", 0.1)]:
        estimates = np.quantile(draws, q, axis=1, method="inverted_cdf")
        result[f"{name}_ci95"] = [
            _number(float(np.quantile(estimates, p, method="inverted_cdf"))) for p in (0.025, 0.975)
        ]
    result["bootstrap_scheme"] = "within_stratum_fixed_counts"
    result["stratum_counts"] = strata.copy()
    return result
