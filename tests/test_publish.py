"""Tests for the publish stage.

Publishing is the one irreversible step in the pipeline, so the rules that
decide WHAT gets uploaded and WHETHER it should be are worth pinning. Nothing
here touches the network — the OAuth and upload paths are thin wrappers over
urllib, while the metadata and gating rules are where the judgment lives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shortform.publish.episode import (
    Episode,
    blocking_reasons,
    build_description,
    build_tags,
)
from shortform.publish.youtube import (
    MAX_DESCRIPTION,
    MAX_TITLE,
    build_metadata,
)


def _episode(**kw) -> Episode:
    base = dict(
        video_id="uburex01e01",
        video_path=Path("/tmp/x.mp4"),
        title="Ubu Rex, Part 1",
        strategy_name="ubu_rex",
        topic="Act 1, scene 1",
    )
    base.update(kw)
    return Episode(**base)  # type: ignore[arg-type]


# --- Description ------------------------------------------------------------


def test_default_description_uses_title_and_topic():
    assert build_description(_episode()).startswith("Ubu Rex, Part 1 — Act 1, scene 1")


def test_strategy_template_wins():
    ep = _episode(publish_config={"description": "{title}. A puppet play about {topic}."})
    assert "A puppet play about Act 1, scene 1." in build_description(ep)


def test_shorts_tag_is_always_appended():
    """Vertical and under three minutes is a Short, but only if labelled — a
    strategy overriding the description must not silently lose it."""
    ep = _episode(publish_config={"description": "Just this."})
    assert build_description(ep).endswith("#Shorts")


def test_shorts_tag_is_not_duplicated():
    ep = _episode(publish_config={"description": "Puppets. #Shorts"})
    assert build_description(ep).count("#Shorts") == 1


def test_footer_is_appended_before_the_tag():
    ep = _episode(publish_config={"description": "Body.", "footer": "Adapted from Jarry."})
    lines = [line for line in build_description(ep).splitlines() if line.strip()]
    assert lines == ["Body.", "Adapted from Jarry.", "#Shorts"]


def test_description_survives_a_missing_topic():
    assert build_description(_episode(topic="")) == "Ubu Rex, Part 1\n\n#Shorts"


# --- Tags -------------------------------------------------------------------


def test_tags_are_deduplicated_in_order():
    ep = _episode(publish_config={"tags": ["ubu", "puppets", "ubu", "claymation"]})
    assert build_tags(ep) == ["ubu", "puppets", "claymation"]


def test_no_tags_is_an_empty_list_not_none():
    assert build_tags(_episode()) == []


def test_blank_tags_are_dropped():
    ep = _episode(publish_config={"tags": ["ubu", "  ", ""]})
    assert build_tags(ep) == ["ubu"]


# --- What blocks an upload --------------------------------------------------


def test_flagged_clips_block_by_default():
    """The critic flags only after exhausting its regenerate ladder, so a flag
    means a human was supposed to look."""
    ep = _episode(flagged=["segment_03_clip_01.mp4"])
    reasons = blocking_reasons(ep)
    assert len(reasons) == 1
    assert "segment_03_clip_01.mp4" in reasons[0]


def test_allow_flagged_overrides():
    ep = _episode(flagged=["segment_03_clip_01.mp4"])
    assert blocking_reasons(ep, allow_flagged=True) == []


def test_unverified_clips_do_not_block():
    """'Never checked' is a weaker signal than 'checked and failed'. Blocking
    on it would stop every upload made without an ANTHROPIC_API_KEY."""
    assert blocking_reasons(_episode(unverified=["segment_02.mp4"])) == []


def test_a_clean_episode_has_no_blockers():
    assert blocking_reasons(_episode()) == []


# --- Upload metadata --------------------------------------------------------


def test_privacy_defaults_to_private():
    assert build_metadata("t", "d", [])["status"]["privacyStatus"] == "private"


def test_over_long_title_is_truncated_not_rejected():
    """A 101-character title should publish with the last character dropped,
    not fail an upload of a video that already cost real money to render."""
    meta = build_metadata("x" * 250, "d", [])
    assert len(meta["snippet"]["title"]) == MAX_TITLE


def test_over_long_description_is_truncated():
    meta = build_metadata("t", "y" * 9000, [])
    assert len(meta["snippet"]["description"]) == MAX_DESCRIPTION


def test_metadata_carries_tags_and_category():
    meta = build_metadata("t", "d", ["a", "b"], category_id="22")
    assert meta["snippet"]["tags"] == ["a", "b"]
    assert meta["snippet"]["categoryId"] == "22"


def test_made_for_kids_is_declared():
    """Required by the API; omitting it leaves the video in limbo until it's
    set manually in Studio."""
    assert build_metadata("t", "d", [])["status"]["selfDeclaredMadeForKids"] is False


# --- Loading an episode -----------------------------------------------------


def test_missing_manifest_is_a_clear_error(tmp_path: Path, monkeypatch):
    from shortform.publish import episode as ep_mod

    monkeypatch.setattr(ep_mod, "VIDEO_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="no manifest"):
        ep_mod.load_episode("nope")


def test_manifest_naming_a_missing_video_is_an_error(tmp_path: Path, monkeypatch):
    """A manifest without its video means a deleted or moved render — better a
    clear error than uploading whatever else is lying around."""
    from shortform.publish import episode as ep_mod

    monkeypatch.setattr(ep_mod, "VIDEO_DIR", tmp_path)
    (tmp_path / "e01.manifest.json").write_text(
        json.dumps({"output": "gone.mp4", "title": "E1"})
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        ep_mod.load_episode("e01")


def test_episode_loads_without_a_script(tmp_path: Path, monkeypatch):
    """`generate-from-script` output has a manifest but may have no script under
    its id — that should degrade, not fail."""
    from shortform.publish import episode as ep_mod

    monkeypatch.setattr(ep_mod, "VIDEO_DIR", tmp_path)
    monkeypatch.setattr(ep_mod, "SCRIPT_DIR", tmp_path / "nothing")
    (tmp_path / "e01.mp4").write_bytes(b"video")
    (tmp_path / "e01.manifest.json").write_text(
        json.dumps({"output": "e01.mp4", "title": "E1", "flagged": ["a.mp4"]})
    )

    ep = ep_mod.load_episode("e01")
    assert ep.title == "E1"
    assert ep.flagged == ["a.mp4"]
    assert ep.strategy_name == ""
