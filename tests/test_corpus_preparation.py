"""Offline archive-to-manifest integration with synthetic FLAC fixtures."""

import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[1]
prepare = runpy.run_path(str(PROJECT / "tools/prepare_s1_corpus.py"))["prepare"]


@pytest.mark.parametrize("subset,takes", [("test-clean", 3), ("dev-other", 1)])
def test_verified_archive_selection_and_failure_guards(tmp_path, subset, takes):
    archive = tmp_path / "test-clean.tar.gz"
    audio = io.BytesIO()
    sf.write(audio, np.zeros(160000), 16000, format="FLAC")
    payload = audio.getvalue()
    with tarfile.open(archive, "w:gz") as bundle:
        for speaker in range(1, 31):
            for take in range(3):
                info = tarfile.TarInfo(f"LibriSpeech/{subset}/{speaker}/1/{speaker}-1-{take}.flac")
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        info = tarfile.TarInfo("LibriSpeech/LICENSE.TXT")
        info.size = 7
        bundle.addfile(info, io.BytesIO(b"fixture"))
        info = tarfile.TarInfo("../../escape.txt")
        info.size = 4
        bundle.addfile(info, io.BytesIO(b"nope"))
    policy = json.loads((PROJECT / "data/manifests/s1_corpus_selection.json").read_text())
    if takes == 1:
        policy.update(
            source_subset=subset,
            scene_mode="disjoint_speakers",
            speaker_count=30,
            utterances_per_speaker=1,
        )
    policy["archive_md5"] = hashlib.md5(archive.read_bytes()).hexdigest()
    policy["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    policy_dir = tmp_path / "data/manifests"
    policy_dir.mkdir(parents=True)
    policy_path = policy_dir / "s1_corpus_selection.json"
    policy_path.write_text(json.dumps(policy))
    code = tmp_path / "src/acoustic_array/io/corpus.py"
    code.parent.mkdir(parents=True)
    code.write_bytes((PROJECT / "src/acoustic_array/io/corpus.py").read_bytes())
    output = tmp_path / "data/raw/selected"
    prepare(archive, output, tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["files"]) == 30 * takes
    assert len(list(output.glob("*.flac"))) == 30 * takes
    assert len(manifest["scenes"]) == (30 if takes == 3 else 10)
    assert not (tmp_path.parent / "escape.txt").exists()
    assert all(r["sha256"] == hashlib.sha256(payload).hexdigest() for r in manifest["files"])
    with pytest.raises(ValueError, match="new"):
        prepare(archive, output, tmp_path)
    with pytest.raises(ValueError):
        prepare(archive, tmp_path / "outside", tmp_path)
    policy["archive_sha256"] = "0" * 64
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="SHA-256"):
        prepare(archive, tmp_path / "data/raw/bad_sha", tmp_path)
    assert not (tmp_path / "data/raw/bad_sha").exists()
    policy["archive_md5"] = "0" * 32
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="MD5"):
        prepare(archive, tmp_path / "data/raw/bad", tmp_path)
    assert not (tmp_path / "data/raw/bad").exists()
