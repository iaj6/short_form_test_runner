"""Tests for the clip continuity critic.

Focus on the decisions that make it safe to run unattended: only fatal issues
regenerate, the escalation ladder drops the chain anchor, and any failure to
check degrades to "passed but unverified" rather than blocking a render.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from shortform.models.script import Segment, Turn
from shortform.visuals.backend import VisualOutput, VisualOutputType
from shortform.visuals.critic import (
    FATAL,
    MINOR,
    ClipCritic,
    Verdict,
    _verdict_from,
    extract_frames,
)


def _clip(path: Path, seconds: float = 2.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=red:s=64x64:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def _png(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=blue:s=64x64:d=0.1", "-frames:v", "1", str(path)],
        check=True, capture_output=True,
    )
    return path


# --- Verdict parsing --------------------------------------------------------


def test_no_issues_passes():
    v = _verdict_from({"issues": [], "summary": "both puppets on model"})
    assert v.passed
    assert v.describe() == "ok"


def test_fatal_issue_fails():
    v = _verdict_from({
        "issues": [{"kind": "character_replaced", "severity": FATAL,
                    "detail": "the woman is now in a blue dress and apron"}],
        "summary": "identity broken",
    })
    assert not v.passed
    assert len(v.fatal_issues) == 1


def test_minor_issue_alone_still_passes():
    """Generational softening is inherent to chaining — flagging it would burn
    credits regenerating clips that are perfectly usable."""
    v = _verdict_from({
        "issues": [{"kind": "quality_degraded", "severity": MINOR,
                    "detail": "backdrop looks over-sharpened"}],
        "summary": "slight drift",
    })
    assert v.passed
    assert v.issues, "the issue is still reported, just not fatal"


def test_mixed_severity_fails_on_the_fatal_one():
    v = _verdict_from({
        "issues": [
            {"kind": "quality_degraded", "severity": MINOR, "detail": "softer"},
            {"kind": "character_missing", "severity": FATAL, "detail": "he is gone"},
        ],
        "summary": "one character lost",
    })
    assert not v.passed
    assert [i.kind for i in v.fatal_issues] == ["character_missing"]


def test_malformed_payload_does_not_crash():
    v = _verdict_from({})
    assert v.passed and v.issues == []


# --- Frame extraction -------------------------------------------------------


def test_extract_frames_samples_across_clip(tmp_path: Path):
    clip = _clip(tmp_path / "c.mp4", seconds=3.0)
    frames = extract_frames(clip, tmp_path / "frames", count=3)
    assert len(frames) == 3
    assert all(f.exists() and f.stat().st_size > 0 for f in frames)


def test_extract_frames_handles_single_sample(tmp_path: Path):
    clip = _clip(tmp_path / "c.mp4", seconds=2.0)
    assert len(extract_frames(clip, tmp_path / "f", count=1)) == 1


def test_extract_frames_returns_empty_for_unreadable(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert extract_frames(bad, tmp_path / "f") == []


# --- Graceful degradation ---------------------------------------------------


def test_no_api_key_passes_as_unverified(tmp_path: Path):
    """A critic that can't run must never block a render."""
    critic = ClipCritic(api_key="")
    v = critic.review(tmp_path / "c.mp4", tmp_path / "ref.png", tmp_path)
    assert v.passed and v.unverified
    assert not critic.available


def test_missing_reference_passes_as_unverified(tmp_path: Path):
    critic = ClipCritic(api_key="sk-test")
    clip = _clip(tmp_path / "c.mp4")
    v = critic.review(clip, tmp_path / "nope.png", tmp_path)
    assert v.passed and v.unverified


def test_api_failure_passes_as_unverified(tmp_path: Path, monkeypatch):
    """A network blip mid-batch must not fail the render."""
    critic = ClipCritic(api_key="sk-test")
    clip = _clip(tmp_path / "c.mp4")
    ref = _png(tmp_path / "ref.png")

    def boom(*a, **k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(critic, "_ask", boom)
    v = critic.review(clip, ref, tmp_path)
    assert v.passed and v.unverified
    assert "connection reset" in v.summary


def test_unverified_is_distinguishable_from_a_real_pass():
    """An unattended batch must not read 'never checked' as 'checked and fine'."""
    real = Verdict(passed=True, summary="ok")
    skipped = Verdict(passed=True, summary="no key", unverified=True)
    assert real.passed and skipped.passed
    assert not real.unverified and skipped.unverified
    assert "unverified" in skipped.describe()


# --- Escalation ladder in VisualGenStage ------------------------------------


class _FakeBackend:
    """Returns video clips and records the config it was called with."""

    name = "fake"

    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[dict[str, Any]] = []
        self._tmp = tmp_path

    async def generate(self, segment, output_path, width, height, config):
        self.calls.append(dict(config))
        path = Path(f"{output_path}.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return VisualOutput(
            path=path, output_type=VisualOutputType.VIDEO, width=width, height=height
        )


class _ScriptedCritic:
    """Fails a fixed number of times, then passes."""

    available = True

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.reviews = 0
        self.seen_actions: list[str] = []

    def review(
        self, clip_path, reference_path, work_dir, expected_characters="",
        intended_action="",
    ):
        self.reviews += 1
        self.seen_actions.append(intended_action)
        if self.reviews <= self.failures:
            return _verdict_from({
                "issues": [{"kind": "character_replaced", "severity": FATAL,
                            "detail": "wrong character"}],
                "summary": "identity broken",
            })
        return _verdict_from({"issues": [], "summary": "ok"})


def _segment() -> Segment:
    return Segment(
        index=0, visual_prompt="x",
        turns=[Turn(speaker="PERE UBU", line="Shitre!")],
    )


@pytest.mark.asyncio
async def test_passing_clip_generates_once(tmp_path: Path):
    from shortform.stages.visual_gen import VisualGenStage

    backend = _FakeBackend(tmp_path)
    stage = VisualGenStage(backend=backend, critic=_ScriptedCritic(failures=0))
    reviews: list[dict[str, Any]] = []

    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=reviews,
    )
    assert len(backend.calls) == 1
    assert reviews[0]["passed"]


@pytest.mark.asyncio
async def test_rejected_clip_is_regenerated(tmp_path: Path):
    from shortform.stages.visual_gen import VisualGenStage

    backend = _FakeBackend(tmp_path)
    stage = VisualGenStage(backend=backend, critic=_ScriptedCritic(failures=1))
    reviews: list[dict[str, Any]] = []

    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=reviews,
    )
    assert len(backend.calls) == 2, "one failure should trigger exactly one retry"
    assert not reviews[0]["passed"] and reviews[1]["passed"]


@pytest.mark.asyncio
async def test_third_attempt_drops_the_chain_anchor(tmp_path: Path):
    """The key escalation: if the frame we chained from is what's broken,
    chaining from it again just reproduces the fault."""
    from shortform.stages.visual_gen import VisualGenStage

    backend = _FakeBackend(tmp_path)
    stage = VisualGenStage(backend=backend, critic=_ScriptedCritic(failures=2))
    reviews: list[dict[str, Any]] = []

    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "hero.png", "chain_from": "lastframe.png"},
        label="test", reference_path="hero.png", work_dir=tmp_path,
        expected_characters="", reviews=reviews,
    )
    assert len(backend.calls) == 3
    assert backend.calls[0]["chain_from"] == "lastframe.png"
    assert backend.calls[1]["chain_from"] == "lastframe.png"
    assert "chain_from" not in backend.calls[2], "attempt 3 re-anchors to the hero"


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts_and_keeps_last(tmp_path: Path):
    """A flagged clip in a finished episode beats a failed render."""
    from shortform.stages.visual_gen import CRITIC_MAX_ATTEMPTS, VisualGenStage

    backend = _FakeBackend(tmp_path)
    stage = VisualGenStage(backend=backend, critic=_ScriptedCritic(failures=99))
    reviews: list[dict[str, Any]] = []

    result = await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=reviews,
    )
    assert len(backend.calls) == CRITIC_MAX_ATTEMPTS
    assert result is not None, "still returns a clip rather than failing the render"
    assert all(not r["passed"] for r in reviews)


@pytest.mark.asyncio
async def test_no_critic_generates_once_and_skips_review(tmp_path: Path):
    """Strategies that don't opt in behave exactly as before."""
    from shortform.stages.visual_gen import VisualGenStage

    backend = _FakeBackend(tmp_path)
    stage = VisualGenStage(backend=backend, critic=None)
    reviews: list[dict[str, Any]] = []

    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={}, label="test", reference_path="", work_dir=tmp_path,
        expected_characters="", reviews=reviews,
    )
    assert len(backend.calls) == 1
    assert reviews == []


# --- Media type detection ---------------------------------------------------
#
# A file's extension is not evidence of its format. A reference image written
# straight from an image API's response bytes carried a .png name while being
# JPEG; the hardcoded media_type produced a 400 that read like a code bug.


def test_detects_png():
    from shortform.visuals.critic import detect_media_type

    assert detect_media_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) == "image/png"


def test_detects_jpeg_regardless_of_extension():
    from shortform.visuals.critic import detect_media_type

    assert detect_media_type(b"\xff\xd8\xff\xe0" + b"\x00" * 32) == "image/jpeg"


def test_detects_webp():
    from shortform.visuals.critic import detect_media_type

    assert detect_media_type(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16) == "image/webp"


def test_unknown_bytes_default_to_png():
    from shortform.visuals.critic import detect_media_type

    assert detect_media_type(b"not an image") == "image/png"


def test_image_block_uses_the_real_type(tmp_path: Path):
    from shortform.visuals.critic import _image_block

    jpeg = tmp_path / "misnamed.png"          # .png name, JPEG bytes
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    assert _image_block(jpeg)["source"]["media_type"] == "image/jpeg"


# --- Scripted action vs. lost character -------------------------------------
#
# The critic compares a clip against a static hero reference, so a stage
# direction that sends a character offstage ("PERE UBU going out, slamming the
# door") reads as `character_missing` — fatal. Ubu Rex e03 segment 1 burned all
# three regenerate attempts on each of three clips, nine Veo calls, and kept
# clips the critic still considered broken. The shot was doing what the script
# asked; the critic just had no way to know that.


def test_staged_action_joins_the_stage_directions():
    seg = Segment(
        index=0,
        turns=[
            Turn(speaker="MERE UBU", line="Oh! Shitre!", stage_direction="aside"),
            Turn(speaker="PERE UBU", line="By my green candle!",
                 stage_direction="going out, slamming the door"),
        ],
    )
    assert seg.staged_action == (
        "MERE UBU aside; PERE UBU going out, slamming the door"
    )


def test_staged_action_is_empty_without_directions():
    """The overwhelmingly common case — no directions, nothing to tell the
    critic, and the prompt gains no INTENDED ACTION section at all."""
    seg = Segment(index=0, turns=[Turn(speaker="MERE UBU", line="Shitre.")])
    assert seg.staged_action == ""


def test_staged_action_skips_turns_that_have_none():
    seg = Segment(
        index=0,
        turns=[
            Turn(speaker="MERE UBU", line="a"),
            Turn(speaker="PERE UBU", line="b", stage_direction="going out"),
            Turn(speaker="MERE UBU", line="c"),
        ],
    )
    assert seg.staged_action == "PERE UBU going out"


def test_single_narrator_segment_has_no_staged_action():
    """Strategies without dialogue carry no turns, so nothing changes for them."""
    assert Segment(index=0, narration="A quiet room.").staged_action == ""


def test_review_forwards_the_intended_action(tmp_path: Path, monkeypatch):
    critic = ClipCritic(api_key="sk-test")
    clip = _clip(tmp_path / "c.mp4")
    ref = _png(tmp_path / "ref.png")
    captured: dict[str, Any] = {}

    def spy(frames, reference_path, expected_characters, intended_action=""):
        captured["action"] = intended_action
        return _verdict_from({"issues": [], "summary": "ok"})

    monkeypatch.setattr(critic, "_ask", spy)
    critic.review(clip, ref, tmp_path, intended_action="PERE UBU going out")
    assert captured["action"] == "PERE UBU going out"


@pytest.mark.asyncio
async def test_stage_hands_the_critic_the_scripted_action(tmp_path: Path):
    """The exit reaches the critic instead of being invisible to it.

    The action is now passed in per clip rather than derived from the whole
    segment — see `build_staged_action`, which decides WHICH clip gets it.
    Deriving it here is what let one exit be expected in all four of a
    segment's clips.
    """
    from shortform.stages.visual_gen import VisualGenStage

    seg = Segment(
        index=1, visual_prompt="x",
        turns=[Turn(speaker="PERE UBU", line="Off I go.",
                    stage_direction="going out, slamming the door")],
    )
    critic = _ScriptedCritic(failures=0)
    stage = VisualGenStage(backend=_FakeBackend(tmp_path), critic=critic)

    await stage._generate_reviewed(
        segment=seg, output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[], intended_action="PERE UBU going out, slamming the door",
    )
    assert critic.seen_actions == ["PERE UBU going out, slamming the door"]


@pytest.mark.asyncio
async def test_a_clip_given_no_action_tells_the_critic_nothing(tmp_path: Path):
    """A clip that covers no annotated turn goes back to strict comparison —
    it must not inherit an exit it isn't supposed to be performing."""
    from shortform.stages.visual_gen import VisualGenStage

    seg = Segment(
        index=1, visual_prompt="x",
        turns=[Turn(speaker="PERE UBU", line="Off I go.",
                    stage_direction="going out, slamming the door")],
    )
    critic = _ScriptedCritic(failures=0)
    stage = VisualGenStage(backend=_FakeBackend(tmp_path), critic=critic)

    await stage._generate_reviewed(
        segment=seg, output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[],
    )
    assert critic.seen_actions == [""]


@pytest.mark.asyncio
async def test_segment_without_directions_sends_no_action(tmp_path: Path):
    from shortform.stages.visual_gen import VisualGenStage

    critic = _ScriptedCritic(failures=0)
    stage = VisualGenStage(backend=_FakeBackend(tmp_path), critic=critic)

    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[],
    )
    assert critic.seen_actions == [""]


def test_blank_frames_is_a_reportable_kind():
    """Rule 4 excuses a scripted exit; rule 5 stops it excusing a black clip.
    The kind needs its own enum slot or the model has to force an empty shot
    into `character_missing` — exactly what rule 4 now waves through."""
    from shortform.visuals.critic import VERDICT_TOOL

    kinds = VERDICT_TOOL["input_schema"]["properties"]["issues"]["items"][
        "properties"
    ]["kind"]["enum"]
    assert "blank_frames" in kinds


def test_blank_frames_verdict_is_fatal():
    v = _verdict_from({
        "issues": [{"kind": "blank_frames", "severity": FATAL,
                    "detail": "all frames solid black"}],
        "summary": "empty shot",
    })
    assert not v.passed


# --- The flagged record's lifecycle -----------------------------------------


@pytest.mark.asyncio
async def test_exhausting_the_ladder_records_the_clip(tmp_path: Path):
    from shortform.stages.visual_gen import VisualGenStage

    stage = VisualGenStage(
        backend=_FakeBackend(tmp_path), critic=_ScriptedCritic(failures=99)
    )
    result = await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[],
    )
    assert stage._flagged == {result.path.name}


@pytest.mark.asyncio
async def test_a_clip_that_now_passes_is_unflagged(tmp_path: Path):
    """Otherwise the record outlives the problem and every later run pays to
    regenerate a clip that has already been fixed."""
    from shortform.stages.visual_gen import VisualGenStage

    stage = VisualGenStage(
        backend=_FakeBackend(tmp_path), critic=_ScriptedCritic(failures=0)
    )
    stage._flagged = {"seg.mp4"}
    result = await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[],
    )
    assert result.path.name == "seg.mp4"
    assert stage._flagged == set()


@pytest.mark.asyncio
async def test_a_clip_passing_only_on_retry_is_not_flagged(tmp_path: Path):
    """The ladder existing to fix things is the point — a clip that needed two
    attempts and then passed is a success, not something to regenerate later."""
    from shortform.stages.visual_gen import VisualGenStage

    stage = VisualGenStage(
        backend=_FakeBackend(tmp_path), critic=_ScriptedCritic(failures=1)
    )
    await stage._generate_reviewed(
        segment=_segment(), output_path=tmp_path / "seg", width=1080, height=1920,
        config={"reference_image": "ref.png"}, label="test",
        reference_path="ref.png", work_dir=tmp_path, expected_characters="",
        reviews=[],
    )
    assert stage._flagged == set()


# --- character_check is load-bearing ----------------------------------------
#
# Two prompt-only attempts to hold the costume boundary failed under real
# conditions. Told a character was performing a scripted exit, the critic
# stopped evaluating that character: e03 passed a clip in which Pere Ubu had
# become a smooth nude featureless pear, with the verdict "Pere Ubu exits
# through the door as scripted while the woman ... remains consistent". A
# required schema field can't be skipped the way a clause in prose can.


def test_costume_mismatch_fails_even_with_no_issues_reported():
    """THE fix: the model must ASSERT the costume is fine, not merely neglect
    to mention that it isn't."""
    v = _verdict_from({
        "issues": [],
        "character_check": [
            {"character": "Pere Ubu", "costume_matches": False,
             "note": "bare, no sleeves or trousers, moustache gone"},
        ],
        "summary": "he exits through the door as scripted",
    })
    assert not v.passed
    assert v.fatal_issues[0].kind == "character_replaced"
    assert "Pere Ubu" in v.fatal_issues[0].detail


def test_costume_ok_does_not_invent_an_issue():
    v = _verdict_from({
        "issues": [],
        "character_check": [
            {"character": "Pere Ubu", "costume_matches": True, "note": "on model"},
            {"character": "Mere Ubu", "costume_matches": True, "note": "on model"},
        ],
        "summary": "ok",
    })
    assert v.passed and v.issues == []


def test_one_bad_costume_among_several_still_fails():
    v = _verdict_from({
        "issues": [],
        "character_check": [
            {"character": "Mere Ubu", "costume_matches": True, "note": "on model"},
            {"character": "Pere Ubu", "costume_matches": False, "note": "nude"},
        ],
        "summary": "she is consistent",
    })
    assert not v.passed
    assert len(v.fatal_issues) == 1


def test_costume_failure_merges_with_reported_issues():
    v = _verdict_from({
        "issues": [{"kind": "quality_degraded", "severity": MINOR, "detail": "soft"}],
        "character_check": [
            {"character": "Pere Ubu", "costume_matches": False, "note": "nude"},
        ],
        "summary": "drifting",
    })
    assert not v.passed
    assert len(v.issues) == 2


def test_missing_character_check_still_parses():
    """Older payloads and malformed responses must not crash — a critic that
    throws is worse than no critic."""
    v = _verdict_from({"issues": [], "summary": "ok"})
    assert v.passed


def test_malformed_character_check_entries_are_ignored():
    v = _verdict_from({
        "issues": [],
        "character_check": ["not a dict", {"character": "X"}, None],
        "summary": "ok",
    })
    assert v.passed


def test_character_check_is_required_by_the_schema():
    """If it's optional the model will omit it for exactly the character the
    action gave it an excuse to skip."""
    from shortform.visuals.critic import VERDICT_TOOL

    assert "character_check" in VERDICT_TOOL["input_schema"]["required"]
    item = VERDICT_TOOL["input_schema"]["properties"]["character_check"]["items"]
    assert set(item["required"]) == {"character", "costume_matches", "note"}
