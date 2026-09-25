"""Selection and scene identity invariants, without external downloads."""

import pytest

from acoustic_array.io.corpus import make_disjoint_scenes, make_s1_scenes, select_s1_records


def records():
    return [
        {
            "utterance_id": f"{s}-10-{u}",
            "speaker_id": str(s),
            "sample_rate_hz": 16000,
            "channels": 1,
            "frames": 160000,
        }
        for s in range(1, 33)
        for u in [11, 2, 1, 10]
    ]


def test_numeric_selection_exclusion_and_order_invariance():
    selected = select_s1_records(records()[::-1], {"1", "2"})
    assert selected == select_s1_records(records(), {"1", "2"})
    assert [r["utterance_id"] for r in selected[:3]] == ["3-10-1", "3-10-2", "3-10-10"]
    assert len(selected) == 90


def test_no_silent_relaxation_and_duplicate_detection():
    with pytest.raises(ValueError, match="Fewer"):
        select_s1_records(records(), {"1", "2", "3"})
    with pytest.raises(ValueError, match="Duplicate"):
        select_s1_records(records() + records()[:1], set())


@pytest.mark.parametrize(
    "field,value", [("frames", 159999), ("channels", 2), ("sample_rate_hz", 48000)]
)
def test_ineligible_audio_not_selected(field, value):
    candidates = records()
    for r in candidates[:4]:
        r[field] = value
    assert select_s1_records(candidates, set())[0]["speaker_id"] == "2"


def test_scenes_have_disjoint_recordings_and_disjoint_speaker_clusters():
    scenes = make_s1_scenes(select_s1_records(records(), set()))
    identifiers = [s["utterance_id"] for scene in scenes for s in scene["sources"]]
    assert len(set(identifiers)) == 90
    speaker_clusters = {}
    speaker_angles = {}
    for scene in scenes:
        assert len({s["speaker_id"] for s in scene["sources"]}) == 3
        for source in scene["sources"]:
            speaker = source["speaker_id"]
            speaker_clusters.setdefault(speaker, set()).add(scene["speaker_cluster_id"])
            speaker_angles.setdefault(speaker, set()).add(source["angle_degrees"])
    assert all(len(c) == 1 for c in speaker_clusters.values())
    assert all(a == {-45, 0, 50} for a in speaker_angles.values())


def test_disjoint_scenes_use_each_speaker_once():
    selected = select_s1_records(records(), set(), utterances_per_speaker=1)
    scenes = make_disjoint_scenes(selected, prefix="test")
    assert len(scenes) == 10
    assert len({s["speaker_id"] for scene in scenes for s in scene["sources"]}) == 30
    assert scenes[0]["sources"][0]["utterance_id"] == "1-10-1"
    assert scenes[0]["sources"][2]["angle_degrees"] == 50
    with pytest.raises(ValueError, match="reused"):
        make_disjoint_scenes(selected[:3] * 2, prefix="bad")
    with pytest.raises(ValueError, match="multiple"):
        make_disjoint_scenes(selected[:2], prefix="bad")
    with pytest.raises(ValueError, match="canonical"):
        make_disjoint_scenes(selected[::-1], prefix="bad")


@pytest.mark.parametrize("count", [0, -1, True, 2.5, 1001])
def test_selection_count_validation(count):
    with pytest.raises(ValueError, match="integers"):
        select_s1_records(records(), set(), speaker_count=count)
