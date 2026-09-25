"""Assemble and verify 30 speaker-disjoint, unscored S1 scenes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from acoustic_array.io.corpus import make_disjoint_scenes, select_s1_records
from acoustic_array.io.dataset_split import validate_splits


def assemble(root: Path, output: Path) -> dict:
    """Verify all source files before writing a new aggregate manifest."""
    output.resolve().relative_to((root / "data/manifests").resolve())
    if output.exists():
        raise ValueError("Refuse to overwrite existing holdout manifest")
    policy_path = root / "data/manifests/s1_disjoint_selection.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    parent_paths = [
        "data/manifests/s1_candidate_v1.json",
        "data/raw/s1_dev_clean_v1/manifest.json",
        "data/raw/s1_dev_other_v1/manifest.json",
    ]
    if policy["subset_order"] != ["test-clean", "dev-clean", "dev-other"]:
        raise ValueError("Unexpected frozen subset order")
    selected, scenes, parents = [], [], []
    for subset, relative in zip(policy["subset_order"], parent_paths, strict=True):
        path = root / relative
        parent = json.loads(path.read_text(encoding="utf-8"))
        if parent["dataset_id"] != policy["dataset_id"]:
            raise ValueError("Parent dataset identity mismatch")
        if parent["status"] != "unscored_holdout_candidate":
            raise ValueError("Parent is not an unscored candidate")
        if parent.get("source_subset", "test-clean") != subset:
            raise ValueError("Parent subset mismatch")
        chosen = select_s1_records(parent["files"], set(), utterances_per_speaker=1)
        selected.extend({**r, "source_subset": subset} for r in chosen)
        scenes.extend(
            {**s, "source_subset": subset}
            for s in make_disjoint_scenes(chosen, prefix=f"s1-{subset}")
        )
        parents.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "archive_sha256": parent["archive_sha256"],
                "source_subset": subset,
            }
        )
    if len({r["speaker_id"] for r in selected}) != 90:
        raise ValueError("Speakers overlap across corpus subsets")
    if len({r["sha256"] for r in selected}) != 90:
        raise ValueError("Source files overlap across corpus subsets")
    for record in selected:
        path = (root / record["path"]).resolve()
        path.relative_to((root / "data/raw").resolve())
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Selected source hash mismatch")
        audio, rate = sf.read(path, dtype="float64")
        if (
            rate != 16000
            or audio.ndim != 1
            or len(audio) != record["frames"]
            or not np.all(np.isfinite(audio))
        ):
            raise ValueError("Selected source decode mismatch")
    registry_path = root / "data/manifests/split_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validation = validate_splits(
        registry["records"]
        + [
            {
                "dataset_id": policy["dataset_id"],
                "speaker_id": r["speaker_id"],
                "sha256": r["sha256"],
                "split": "heldout",
            }
            for r in selected
        ]
    )
    result = {
        "schema_version": "1.0",
        "status": "unscored_speaker_disjoint_holdout",
        "dataset_id": policy["dataset_id"],
        "license": policy["license"],
        "attribution": (
            "LibriSpeech: Panayotov, Chen, Povey and Khudanpur (ICASSP 2015); LibriVox readers."
        ),
        "source_page": "https://www.openslr.org/12/",
        "modifications": (
            "Original FLAC unchanged. Evaluation will crop 0..10 s and resample 16 to 48 kHz."
        ),
        "files": selected,
        "scenes": scenes,
        "parents": parents,
        "statistical_scope": policy["statistical_scope"],
        "split_validation": validation,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_code_sha256": hashlib.sha256(
            (root / "src/acoustic_array/io/corpus.py").read_bytes()
        ).hexdigest(),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified 90 speakers / 90 recordings / 30 disjoint scenes: {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assemble(Path(__file__).resolve().parents[1], args.output.resolve())
