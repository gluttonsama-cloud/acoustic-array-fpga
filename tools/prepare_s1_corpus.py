"""Verify one official archive and extract a policy-selected bounded subset.

No network, tar extraction, algorithm evaluation or source-audio modifications.
Run with a local test-clean.tar.gz and a new output directory.
"""

import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import soundfile as sf

from acoustic_array.io.corpus import make_disjoint_scenes, make_s1_scenes, select_s1_records


def prepare(archive: Path, output: Path, root: Path, policy_path: Path | None = None) -> None:
    policy_path = policy_path or root / "data/manifests/s1_corpus_selection.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    subset = policy.get("source_subset", "test-clean")
    if subset not in {"test-clean", "test-other", "dev-clean", "dev-other"}:
        raise ValueError("Unsupported corpus subset")
    mode = policy.get("scene_mode", "repeated_speakers")
    speaker_count = policy.get("speaker_count", 30)
    takes = policy.get("utterances_per_speaker", 3)
    if any(type(n) is not int or not 1 <= n <= 1000 for n in (speaker_count, takes)):
        raise ValueError("Selection counts must be integers in [1, 1000]")
    if mode not in {"repeated_speakers", "disjoint_speakers"}:
        raise ValueError("Unknown scene mode")
    if mode == "repeated_speakers" and (speaker_count, takes) != (30, 3):
        raise ValueError("Legacy scene mode requires 30 speakers and 3 takes")
    if mode == "disjoint_speakers" and (takes != 1 or speaker_count % 3):
        raise ValueError("Disjoint mode requires one take and speaker count divisible by 3")
    if output.exists():
        raise ValueError("Output must be new; preserve existing evidence")
    output.resolve().relative_to((root / "data/raw").resolve())
    if archive.stat().st_size > policy["max_archive_bytes"]:
        raise ValueError("Archive exceeds size limit")
    with archive.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != policy["archive_md5"]:
        raise ValueError("Official archive MD5 mismatch")
    with archive.open("rb") as stream:
        archive_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive_sha != policy["archive_sha256"]:
        raise ValueError("Pinned archive SHA-256 mismatch")
    pattern = re.compile(rf"LibriSpeech/{re.escape(subset)}/(\d+)/(\d+)/(\d+-\d+-\d+)\.flac")
    records = []
    metadata = {}
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            match = pattern.fullmatch(member.name)
            if not match and member.name not in {
                "LibriSpeech/LICENSE.TXT",
                "LibriSpeech/README.TXT",
                "LibriSpeech/SPEAKERS.TXT",
            }:
                continue
            if member.size > 10_000_000:
                raise ValueError("Member exceeds size limit")
            stream = bundle.extractfile(member)
            assert stream is not None
            data = stream.read()
            if not match:
                metadata[Path(member.name).name] = data
                continue
            speaker, chapter, identifier = match.groups()
            if identifier.split("-")[:2] != [speaker, chapter]:
                raise ValueError("Archive path and identity disagree")
            info = sf.info(io.BytesIO(data))
            records.append(
                {
                    "utterance_id": identifier,
                    "speaker_id": speaker,
                    "sample_rate_hz": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "archive_member": member.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    if "LICENSE.TXT" not in metadata:
        raise ValueError("Archive license missing")
    selected = select_s1_records(
        records,
        set(policy["excluded_speakers"]),
        speaker_count=speaker_count,
        utterances_per_speaker=takes,
    )
    scenes = (
        make_disjoint_scenes(selected, prefix=f"s1-{subset}")
        if mode == "disjoint_speakers"
        else make_s1_scenes(selected)
    )
    if sum(r["bytes"] for r in selected) > policy["max_selected_bytes"]:
        raise ValueError("Selected corpus exceeds size limit")
    wanted = {r["archive_member"]: r for r in selected}
    output.mkdir(parents=True)
    # Direct regular-file writes with validated flat identifiers, never extractall.
    with tarfile.open(archive, "r|gz") as bundle:
        for member in bundle:
            if member.name not in wanted:
                continue
            record = wanted.pop(member.name)
            stream = bundle.extractfile(member)
            assert stream is not None
            data = stream.read()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Archive changed during preparation")
            destination = output / f"{record['utterance_id']}.flac"
            destination.write_bytes(data)
            record["path"] = destination.relative_to(root).as_posix()
    if wanted:
        raise ValueError("Missing selected archive members")
    for name, data in metadata.items():
        (output / name).write_bytes(data)
    manifest = {
        "schema_version": "1.0",
        "source_subset": subset,
        "scene_mode": mode,
        "status": "unscored_holdout_candidate",
        "dataset_id": policy["dataset_id"],
        "license": policy["license"],
        "attribution": (
            "LibriSpeech: Panayotov, Chen, Povey and Khudanpur, ICASSP 2015; LibriVox readers."
        ),
        "source_page": policy["source_page"],
        "archive_url": policy["archive_url"],
        "archive_md5": md5,
        "archive_sha256": archive_sha,
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_code_sha256": hashlib.sha256(
            (root / "src/acoustic_array/io/corpus.py").read_bytes()
        ).hexdigest(),
        "soundfile_version": sf.__version__,
        "files": selected,
        "scenes": scenes,
        "statistical_unit": policy["statistical_unit"],
        "modifications": (
            "FLAC unchanged. Planned evaluation crops first 10 s and resamples 16 to 48 kHz; "
            "no new high-frequency content."
        ),
        "metadata_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in metadata.items()},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(selected)} recordings, {len(scenes)} scenes: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    prepare(args.archive.resolve(), args.output.resolve(), project, args.policy)
