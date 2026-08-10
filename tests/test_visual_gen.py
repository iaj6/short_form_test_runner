"""Tests for visual_gen helpers — per-segment camera move injection (#3)."""

from __future__ import annotations

from pathlib import Path

from shortform.models.script import Segment, Turn, TurnTiming
from shortform.stages.visual_gen import _apply_camera_move


def test_camera_move_noop_without_moves():
    cfg = {"animation_style": "stop-motion claymation"}
    _apply_camera_move(cfg, 0)
    assert cfg["animation_style"] == "stop-motion claymation"


def test_camera_move_prepends_and_cycles():
    base = "stop-motion claymation, candlelight"
    moves = ["push-in", "dolly-out", "locked shot"]
    styles = []
    for i in range(4):
        cfg = {"animation_style": base, "camera_moves": moves}
        _apply_camera_move(cfg, i)
        styles.append(cfg["animation_style"])

    # Each segment gets a distinct move prepended to the base look.
    assert styles[0] == "push-in, " + base
    assert styles[1] == "dolly-out, " + base
    assert styles[2] == "locked shot, " + base
    # Cycles back around (index 3 % 3 == 0).
    assert styles[3] == "push-in, " + base


def test_camera_move_without_base_style():
    cfg = {"camera_moves": ["slow pan"]}
    _apply_camera_move(cfg, 0)
    assert cfg["animation_style"] == "slow pan"


# --- Clip reuse (resume after an interrupted Veo run) ----------------------


def _stage(reuse: bool = True):
    """A stage with reuse already resolved for the run.

    `_cached` is gated on per-run provenance (see _reuse_allowed); these tests
    exercise `_cached` itself, so the gate is set directly.
    """
    from shortform.stages.visual_gen import VisualGenStage

    stage = VisualGenStage(backend=None, reuse_existing=reuse)
    stage._reuse_this_run = reuse
    return stage


def _real_mp4(path: Path, seconds: float = 1.0) -> Path:
    import subprocess

    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=black:s=64x64:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path


def test_reuse_finds_existing_video(tmp_path: Path):
    from shortform.visuals.backend import VisualOutputType

    _real_mp4(tmp_path / "segment_00.mp4")
    got = _stage()._cached(tmp_path / "segment_00", 1080, 1920)
    assert got is not None
    assert got.output_type == VisualOutputType.VIDEO
    assert got.path.name == "segment_00.mp4"


def test_reuse_finds_existing_still(tmp_path: Path):
    """Veo's safety-filter fallback writes a PNG — that counts as done too."""
    from shortform.visuals.backend import VisualOutputType

    (tmp_path / "segment_00.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    got = _stage()._cached(tmp_path / "segment_00", 1080, 1920)
    assert got is not None
    assert got.output_type == VisualOutputType.IMAGE


def test_reuse_rejects_truncated_video(tmp_path: Path):
    """A run killed mid-download leaves a plausible but unreadable MP4.
    Reusing it would break an episode far downstream."""
    (tmp_path / "segment_00.mp4").write_bytes(b"\x00\x00\x00 ftypisom" + b"\x00" * 4096)
    assert _stage()._cached(tmp_path / "segment_00", 1080, 1920) is None


def test_reuse_ignores_empty_file(tmp_path: Path):
    (tmp_path / "segment_00.mp4").write_bytes(b"")
    assert _stage()._cached(tmp_path / "segment_00", 1080, 1920) is None


def test_reuse_returns_none_when_missing(tmp_path: Path):
    assert _stage()._cached(tmp_path / "segment_00", 1080, 1920) is None


def test_regenerate_flag_disables_reuse(tmp_path: Path):
    """--regenerate must bypass a perfectly valid cached clip."""
    _real_mp4(tmp_path / "segment_00.mp4")
    assert _stage(reuse=False)._cached(tmp_path / "segment_00", 1080, 1920) is None


# --- Speech schedule (dialogue -> video model) ------------------------------


def _timed_segment() -> Segment:
    """A segment whose turns straddle the 7.5s clip boundary."""

    seg = Segment(
        index=0,
        visual_prompt="x",
        turns=[
            Turn(speaker="PERE UBU", line="Shitre!"),
            Turn(speaker="MERE UBU", line="You are a right great blackguard."),
            Turn(speaker="PERE UBU", line="Why do I not brain you!"),
        ],
    )
    seg.turn_timings = [
        TurnTiming(speaker="PERE UBU", start=0.0, duration=1.2),
        TurnTiming(speaker="MERE UBU", start=1.5, duration=5.0),   # crosses 7.5s? no
        TurnTiming(speaker="PERE UBU", start=7.0, duration=3.0),   # straddles the boundary
    ]
    return seg


DESCRIPTIONS = {
    "pere_ubu": "the short fat man in the red hood",
    "mere_ubu": "the tall thin woman in the green shawl",
}


def test_speech_schedule_uses_physical_descriptions():
    """The video model has never heard of 'PERE UBU'."""
    from shortform.stages.visual_gen import build_speech_schedule

    text = build_speech_schedule(_timed_segment(), 0, 7.5, DESCRIPTIONS)
    assert "the short fat man in the red hood" in text
    assert "PERE UBU" not in text


def test_speech_schedule_only_covers_this_clips_window():
    """Clip 1 must not be told about dialogue that already happened in clip 0."""
    from shortform.stages.visual_gen import build_speech_schedule

    seg = _timed_segment()
    clip0 = build_speech_schedule(seg, 0, 7.5, DESCRIPTIONS)
    clip1 = build_speech_schedule(seg, 1, 7.5, DESCRIPTIONS)

    # Turn 1 (1.5-6.5s) belongs to clip 0 only.
    assert "green shawl" in clip0
    assert "green shawl" not in clip1
    # Turn 2 straddles 7.5s, so it appears in both, re-based per clip.
    assert "red hood" in clip0 and "red hood" in clip1
    # Clip 1 times are relative to clip 1's start, not absolute.
    assert "from 0.0s" in clip1


def test_speech_schedule_forbids_simultaneous_mouths():
    """The instruction that actually fixes the look."""
    from shortform.stages.visual_gen import build_speech_schedule

    text = build_speech_schedule(_timed_segment(), 0, 7.5, DESCRIPTIONS)
    assert "keeps its mouth firmly closed" in text
    assert "Never move two mouths at once." in text


def test_speech_schedule_empty_for_narration():
    """Single-narrator segments have no speaker to show — no schedule."""
    from shortform.stages.visual_gen import build_speech_schedule

    seg = Segment(index=0, narration="Bartholomew read the email.", visual_prompt="x")
    assert build_speech_schedule(seg, 0, 7.5, DESCRIPTIONS) == ""


def test_speech_schedule_empty_when_clip_has_no_dialogue():
    """A clip past the end of the dialogue gets no schedule rather than a stale one."""
    from shortform.stages.visual_gen import build_speech_schedule

    assert build_speech_schedule(_timed_segment(), 5, 7.5, DESCRIPTIONS) == ""


def test_veo_prompt_appends_schedule_as_its_own_sentence():
    """Buried in the comma-joined descriptor list, the instruction got ignored."""
    from shortform.visuals.veo_backend import _build_animation_prompt

    prompt = _build_animation_prompt("a room", "claymation", "Dialogue timing: x speaks.")
    assert prompt.startswith("claymation, a room, smooth motion, high quality.")
    assert prompt.endswith("Dialogue timing: x speaks.")
    # Unchanged for narration strategies.
    assert _build_animation_prompt("a room", "claymation") == (
        "claymation, a room, smooth motion, high quality"
    )


# --- Backend-aware clip reuse -----------------------------------------------
#
# A Veo run resuming into a directory left by a cheap Pillow pass silently
# reused the stills, called Veo zero times, and reported success — and the
# batch runner then wrote a manifest claiming the episode WAS rendered with
# Veo, so the lie persisted and every later run skipped it.


def _stage_for(backend_name: str):
    from shortform.stages.visual_gen import VisualGenStage

    class _Backend:
        name = backend_name

        async def generate(self, **kw):  # pragma: no cover - not called here
            raise AssertionError("not used")

    return VisualGenStage(backend=_Backend())


def test_reuse_allowed_when_backend_matches(tmp_path: Path):
    from shortform.stages.visual_gen import PROVENANCE_FILE

    (tmp_path / PROVENANCE_FILE).write_text("veo")
    assert _stage_for("veo")._reuse_allowed(tmp_path) is True


def test_reuse_denied_when_backend_differs(tmp_path: Path):
    """THE bug: a pillow pass must not satisfy a veo run."""
    from shortform.stages.visual_gen import PROVENANCE_FILE

    (tmp_path / PROVENANCE_FILE).write_text("pillow")
    (tmp_path / "segment_00.png").write_bytes(b"still")
    assert _stage_for("veo")._reuse_allowed(tmp_path) is False


def test_reuse_denied_when_provenance_unknown(tmp_path: Path):
    """Clips with no recorded backend predate the check — regenerate rather
    than assume, since a wrong reuse ships the wrong video silently."""
    (tmp_path / "segment_00.png").write_bytes(b"still")
    assert _stage_for("veo")._reuse_allowed(tmp_path) is False


def test_empty_directory_is_reusable(tmp_path: Path):
    """Nothing to reuse, but nothing to warn about either — a fresh run."""
    assert _stage_for("veo")._reuse_allowed(tmp_path) is True


def test_regenerate_flag_overrides_matching_provenance(tmp_path: Path):
    from shortform.stages.visual_gen import PROVENANCE_FILE, VisualGenStage

    (tmp_path / PROVENANCE_FILE).write_text("veo")

    class _Backend:
        name = "veo"

        async def generate(self, **kw):  # pragma: no cover
            raise AssertionError("not used")

    stage = VisualGenStage(backend=_Backend(), reuse_existing=False)
    assert stage._reuse_allowed(tmp_path) is False


def test_provenance_is_recorded_for_the_resume(tmp_path: Path):
    """Written at the START of a run, so an interrupted one still resumes into
    its own clips."""
    from shortform.stages.visual_gen import PROVENANCE_FILE

    stage = _stage_for("veo")
    stage._record_provenance(tmp_path)
    assert (tmp_path / PROVENANCE_FILE).read_text() == "veo"
    assert stage._reuse_allowed(tmp_path) is True
