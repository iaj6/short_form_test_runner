"""Tests for sound-effect cues.

The subtle one is timing: an effect inserted mid-segment pushes every later
turn back, and if the turn timings don't account for it the video's speech
schedule drifts out of step with the audio — the wrong puppet's mouth moves for
the rest of the segment, with nothing in the logs to say why.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from shortform.models.script import Script, Segment, Turn
from shortform.sfx import load_library, resolve
from shortform.tts.concat import SFX_LEAD_GAP, build_clip_plan, concat_turn_audio


def _library(tmp_path: Path, names: list[str], make_files: bool = True) -> Path:
    d = tmp_path / "sfx"
    d.mkdir(parents=True, exist_ok=True)
    effects = []
    for n in names:
        if make_files:
            (d / f"{n}.mp3").write_bytes(b"audio")
        effects.append({"name": n, "file": f"{n}.mp3", "description": n})
    (d / "library.yaml").write_text(yaml.safe_dump({"effects": effects}))
    return d


def _tone(path: Path, seconds: float) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}:sample_rate=44100",
         "-ac", "1", "-c:a", "libmp3lame", str(path)],
        check=True, capture_output=True,
    )
    return path


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


@pytest.fixture(autouse=True)
def _clear_cache():
    """load_library is cached for the run; tests build fresh libraries."""
    load_library.cache_clear()
    yield
    load_library.cache_clear()


# --- Library ----------------------------------------------------------------


def test_resolves_a_known_cue(tmp_path: Path):
    d = _library(tmp_path, ["door_slam", "body_thump"])
    assert resolve("door_slam", d) == d / "door_slam.mp3"


def test_unknown_cue_returns_none(tmp_path: Path):
    """A missing effect degrades an episode; failing the render throws away
    everything already generated for it, which is much worse."""
    d = _library(tmp_path, ["door_slam"])
    assert resolve("nope", d) is None


def test_cue_without_audio_is_skipped(tmp_path: Path):
    """Manifest is committed, audio is gitignored — a fresh clone has cues with
    nothing behind them yet."""
    d = _library(tmp_path, ["door_slam"], make_files=False)
    assert resolve("door_slam", d) is None


def test_empty_name_resolves_to_none(tmp_path: Path):
    assert resolve("", _library(tmp_path, ["door_slam"])) is None


def test_missing_manifest_is_not_fatal(tmp_path: Path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    assert load_library(empty) == {}
    assert resolve("door_slam", empty) is None


# --- Clip plan --------------------------------------------------------------


def test_plan_without_effects_is_one_clip_per_turn():
    plan = build_clip_plan([Path("a.mp3"), Path("b.mp3")], gaps=[0.3, 0.3])
    assert [p.name for p, _ in plan] == ["a.mp3", "b.mp3"]
    assert plan[-1][1] == 0.0, "no trailing pad — assembly owns inter-segment spacing"


def test_effect_is_inserted_after_the_line_it_annotates():
    plan = build_clip_plan(
        [Path("a.mp3"), Path("b.mp3")],
        gaps=[0.85, 0.3],
        gap_audio=[Path("slam.mp3"), None],
    )
    assert [p.name for p, _ in plan] == ["a.mp3", "slam.mp3", "b.mp3"]
    # A short beat so the effect doesn't clip the tail of the speech...
    assert plan[0][1] == SFX_LEAD_GAP
    # ...then the stage-direction gap the line already earned.
    assert plan[1][1] == 0.85


def test_effect_on_the_last_turn_gets_no_trailing_pad():
    plan = build_clip_plan(
        [Path("a.mp3")], gaps=[0.85], gap_audio=[Path("slam.mp3")]
    )
    assert [p.name for p, _ in plan] == ["a.mp3", "slam.mp3"]
    assert plan[-1][1] == 0.0


# --- Real audio -------------------------------------------------------------


def test_effect_lands_in_the_gap(tmp_path: Path):
    line = _tone(tmp_path / "line.mp3", 1.0)
    nxt = _tone(tmp_path / "next.mp3", 1.0)
    slam = _tone(tmp_path / "slam.mp3", 2.0)
    out = tmp_path / "joined.mp3"

    concat_turn_audio(
        [line, nxt], out, gaps=[0.85, 0.0], gap_audio=[slam, None]
    )
    # line(1.0) + lead(0.15) + slam(2.0) + gap(0.85) + next(1.0)
    assert _duration(out) == pytest.approx(5.0, abs=0.1)


def test_no_effects_matches_previous_behaviour(tmp_path: Path):
    a = _tone(tmp_path / "a.mp3", 1.0)
    b = _tone(tmp_path / "b.mp3", 1.0)
    out = tmp_path / "joined.mp3"
    concat_turn_audio([a, b], out, gaps=[0.5, 0.0])
    assert _duration(out) == pytest.approx(2.5, abs=0.05)


# --- Timing sync (the bug worth catching) -----------------------------------


@pytest.mark.asyncio
async def test_turn_timings_account_for_effect_duration(tmp_path: Path, monkeypatch):
    """Without this the speech schedule drifts: the video keeps animating the
    turn the model *thinks* is playing, silently, for the rest of the segment."""
    from shortform.stages import tts as tts_stage
    from tests.test_dialogue import _dialogue_ctx, _FakeBackend, _init_store

    effect = _tone(tmp_path / "slam.mp3", 2.0)

    seg = Segment(
        index=0,
        visual_prompt="x",
        turns=[
            Turn(speaker="PERE UBU", line="One.", sfx="door_slam"),
            Turn(speaker="MERE UBU", line="Two."),
        ],
    )
    ctx = _dialogue_ctx(tmp_path, [seg])
    ctx.strategy.tts["voices"] = {"pere_ubu": {}, "mere_ubu": {}}

    backend = _FakeBackend("f5_tts")  # every turn reports duration 1.0
    monkeypatch.setattr(tts_stage, "get_backend", lambda name, **kw: backend)
    monkeypatch.setattr(tts_stage, "resolve_sfx", lambda name: effect)
    monkeypatch.setattr(
        tts_stage, "concat_turn_audio",
        lambda inputs, output, **kw: output.write_bytes(b"j"),
    )
    monkeypatch.setattr(tts_stage, "get_audio_duration", lambda p: 2.0)
    monkeypatch.setattr(
        tts_stage.FileStore, "__init__",
        lambda self, base_dir=None: _init_store(self, tmp_path),
    )

    await tts_stage.TTSStage().execute(ctx)

    gap = ctx.strategy.tts["turn_gap"]
    # turn 0 at 0.0; turn 1 after turn0(1.0) + gap + lead(0.15) + effect(2.0)
    assert seg.turn_timings[0].start == pytest.approx(0.0)
    assert seg.turn_timings[1].start == pytest.approx(1.0 + gap + SFX_LEAD_GAP + 2.0)


# --- Persistence ------------------------------------------------------------


def test_sfx_round_trips_through_script_json(tmp_path: Path):
    script = Script(
        strategy_name="s",
        segments=[
            Segment(
                index=0, visual_prompt="x",
                turns=[
                    Turn(speaker="A", line="Out.", stage_direction="slamming the door",
                         sfx="door_slam"),
                    Turn(speaker="B", line="Alone."),
                ],
            )
        ],
    )
    path = tmp_path / "s.json"
    script.save_json(path)
    turns = Script.load_json(path).segments[0].turns

    assert turns[0].sfx == "door_slam"
    assert turns[1].sfx == "", "turns without a cue stay clean"
