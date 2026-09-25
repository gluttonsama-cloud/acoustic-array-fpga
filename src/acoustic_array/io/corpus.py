"""Deterministic corpus selection, independent of beamformer scores."""


def select_s1_records(
    records: list[dict],
    excluded: set[str],
    *,
    speaker_count: int = 30,
    utterances_per_speaker: int = 3,
) -> list[dict]:
    """Select a bounded number of speakers and >=10 s recordings by numeric ID."""
    if any(
        type(n) is not int or not 1 <= n <= 1000 for n in (speaker_count, utterances_per_speaker)
    ):
        raise ValueError("Selection counts must be integers in [1, 1000]")
    grouped: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for record in records:
        identifier = record["utterance_id"]
        parts = identifier.split("-")
        if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
            raise ValueError("Invalid LibriSpeech utterance identity")
        if parts[0] != record["speaker_id"] or identifier in seen:
            raise ValueError("Duplicate or inconsistent utterance identity")
        seen.add(identifier)
        if (
            record["speaker_id"] not in excluded
            and record["sample_rate_hz"] == 16000
            and record["channels"] == 1
            and record["frames"] >= 160000
        ):
            grouped.setdefault(record["speaker_id"], []).append(record)
    speakers = sorted((s for s in grouped if len(grouped[s]) >= utterances_per_speaker), key=int)
    if len(speakers) < speaker_count:
        raise ValueError(f"Fewer than {speaker_count} eligible speakers; do not relax selection")
    return [
        record
        for speaker in speakers[:speaker_count]
        for record in sorted(
            grouped[speaker], key=lambda r: tuple(map(int, r["utterance_id"].split("-")))
        )[:utterances_per_speaker]
    ]


def make_disjoint_scenes(selected: list[dict], *, prefix: str) -> list[dict]:
    """One occurrence per speaker; independence remains conditional on the corpus."""
    if not selected or len(selected) % 3:
        raise ValueError("Expected a positive multiple of three recordings")
    if len({r["speaker_id"] for r in selected}) != len(selected):
        raise ValueError("Speaker reused across disjoint scenes")
    canonical = select_s1_records(
        selected, set(), speaker_count=len(selected), utterances_per_speaker=1
    )
    if [r["utterance_id"] for r in selected] != [r["utterance_id"] for r in canonical]:
        raise ValueError("Selected recordings must be in canonical order")
    return [
        {
            "scene_id": f"{prefix}-{i // 3:02d}",
            "speaker_cluster_id": f"{prefix}-{i // 3:02d}",
            "sources": [
                {
                    "utterance_id": r["utterance_id"],
                    "speaker_id": r["speaker_id"],
                    "angle_degrees": angle,
                    "start_seconds": 0,
                    "duration_seconds": 10,
                }
                for r, angle in zip(selected[i : i + 3], [-45, 0, 50], strict=True)
            ],
        }
        for i in range(0, len(selected), 3)
    ]


def make_s1_scenes(selected: list[dict]) -> list[dict]:
    """Thirty unique-recording scenes in ten disjoint speaker clusters."""
    if len(selected) != 90:
        raise ValueError("Expected 90 selected recordings")
    canonical = select_s1_records(selected, set())
    if [r["utterance_id"] for r in selected] != [r["utterance_id"] for r in canonical]:
        raise ValueError("Selected recordings must be in canonical order")
    scenes = []
    for cluster in range(10):
        for take in range(3):
            sources = []
            for position, angle in enumerate([-45, 0, 50]):
                speaker_slot = (position + take) % 3
                record = selected[cluster * 9 + speaker_slot * 3 + take]
                sources.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "speaker_id": record["speaker_id"],
                        "angle_degrees": angle,
                        "start_seconds": 0,
                        "duration_seconds": 10,
                    }
                )
            scenes.append(
                {
                    "scene_id": f"s1-c{cluster:02d}-t{take}",
                    "speaker_cluster_id": f"c{cluster:02d}",
                    "sources": sources,
                }
            )
    return scenes
