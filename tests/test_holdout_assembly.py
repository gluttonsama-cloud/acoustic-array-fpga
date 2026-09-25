"""Exercise aggregate provenance and cross-subset leakage without downloads."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
assemble = runpy.run_path(str(PROJECT / "tools/assemble_s1_holdout.py"))["assemble"]


def test_assembly_integrity_and_leakage(tmp_path):
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    policy = PROJECT / "data/manifests/s1_disjoint_selection.json"
    (manifests / policy.name).write_bytes(policy.read_bytes())
    registry = manifests / "split_registry.json"
    registry.write_text('{"records": []}')
    code = tmp_path / "src/acoustic_array/io/corpus.py"
    code.parent.mkdir(parents=True)
    code.write_bytes((PROJECT / "src/acoustic_array/io/corpus.py").read_bytes())
    parent_paths = [
        manifests / "s1_candidate_v1.json",
        tmp_path / "data/raw/s1_dev_clean_v1/manifest.json",
        tmp_path / "data/raw/s1_dev_other_v1/manifest.json",
    ]
    all_records = []
    for index, (subset, parent) in enumerate(
        zip(["test-clean", "dev-clean", "dev-other"], parent_paths, strict=True)
    ):
        parent.parent.mkdir(parents=True, exist_ok=True)
        folder = tmp_path / f"data/raw/{subset}"
        folder.mkdir(parents=True)
        records = []
        for speaker in range(index * 30 + 1, index * 30 + 31):
            audio_path = folder / f"{speaker}-1-0.flac"
            sf.write(audio_path, np.full(160000, speaker / 1000), 16000)
            records.append(
                {
                    "speaker_id": str(speaker),
                    "utterance_id": f"{speaker}-1-0",
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "frames": 160000,
                    "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    "path": audio_path.relative_to(tmp_path).as_posix(),
                }
            )
        all_records.extend(records)
        parent.write_text(
            json.dumps(
                {
                    "dataset_id": "openslr12-librispeech",
                    "source_subset": subset,
                    "status": "unscored_holdout_candidate",
                    "files": records,
                    "archive_sha256": "1" * 64,
                }
            )
        )
    output = manifests / "aggregate.json"
    result = assemble(tmp_path, output)
    assert len(result["scenes"]) == 30
    assert len({r["speaker_id"] for r in result["files"]}) == 90
    with pytest.raises(ValueError, match="overwrite"):
        assemble(tmp_path, output)
    # Corrupted audio must fail before writing a manifest.
    first_audio = tmp_path / all_records[0]["path"]
    original = first_audio.read_bytes()
    first_audio.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        assemble(tmp_path, manifests / "bad_hash.json")
    assert not (manifests / "bad_hash.json").exists()
    first_audio.write_bytes(original)
    registry.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "dataset_id": "openslr12-librispeech",
                        "speaker_id": "1",
                        "sha256": "2" * 64,
                        "split": "development",
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="leakage"):
        assemble(tmp_path, manifests / "bad_split.json")
    assert not (manifests / "bad_split.json").exists()
    registry.write_text('{"records": []}')
    second = json.loads(parent_paths[1].read_text())
    second["files"][0] = all_records[0]
    parent_paths[1].write_text(json.dumps(second))
    with pytest.raises(ValueError, match="overlap"):
        assemble(tmp_path, manifests / "bad_overlap.json")
