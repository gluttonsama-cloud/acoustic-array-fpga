"""Explicit speaker/file split leakage checks; does not certify scene independence."""

import re


def validate_splits(records: list[dict]) -> dict:
    speakers = {}
    files = {}
    counts = {"development": 0, "tuning": 0, "heldout": 0}
    for record in records:
        split = record["split"]
        if split not in counts:
            raise ValueError("Unknown dataset split")
        for key in ["dataset_id", "speaker_id"]:
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError("Invalid identity")
        digest = record["sha256"]
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("Invalid sha256")
        speaker = (record["dataset_id"].strip(), record["speaker_id"].strip())
        if speaker in speakers and speakers[speaker] != split:
            raise ValueError("Speaker leakage across splits")
        if digest in files and files[digest] != split:
            raise ValueError("Source file leakage across splits")
        speakers[speaker] = split
        files[digest] = split
        counts[split] += 1
    return {
        "status": "split_overlap_check_passed",
        "record_counts": counts,
        "unique_speakers": len(speakers),
        "unique_files": len(files),
        "has_heldout": counts["heldout"] > 0,
    }
