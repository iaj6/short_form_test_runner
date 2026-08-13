"""Tests for multi-speaker dialogue segments.

Covers the three pieces that make an adapted play work: the Turn model and its
derived narration, per-speaker voice resolution (including mixed backends), and
turn-audio concatenation.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from shortform.models.script import (
    Script,
    Segment,
    Turn,
    WordTiming,
    estimate_segment_duration,
)
from shortform.stages.tts import _merge_turn_timings
from shortform.tts.backend import TTSOutput
from shortform.tts.cast import UnknownSpeakerError, VoiceCast, normalize_speaker
from shortform.tts.concat import concat_turn_audio

# --- Turn model ------------------------------------------------------------


def test_narration_derived_from_turns():
    """Every existing .narration consumer keeps working on a dialogue segment."""
    seg = Segment(
        index=0,
        visual_prompt="two clay figures on a bare road",
        turns=[
            Turn(speaker="PERE UBU", line="Shittr!"),
            Turn(speaker="MERE UBU", line="Oh! what a foul word."),
        ],
    )
    assert seg.narration == "Shittr! Oh! what a foul word."
    assert seg.is_dialogue
    assert seg.speakers == ["PERE UBU", "MERE UBU"]


def test_explicit_narration_wins_over_derived():
    """An adaptation can hand-write a cleaner flat text than the naive join."""
    seg = Segment(
        index=0,
        narration="A hand-written summary line.",
        turns=[Turn(speaker="a", line="one"), Turn(speaker="b", line="two")],
    )
    assert seg.narration == "A hand-written summary line."


def test_stage_directions_are_not_spoken():
    seg = Segment(
        index=0,
        turns=[Turn(speaker="ubu", line="I shall kill him.", stage_direction="aside")],
    )
    assert seg.narration == "I shall kill him."
    assert "aside" not in seg.narration


def test_single_narrator_segment_unchanged():
    """The original shape still behaves exactly as before."""
    seg = Segment(index=0, narration="Bartholomew read the email.", visual_prompt="x")
    assert not seg.is_dialogue
    assert seg.speakers == []
    assert seg.narration == "Bartholomew read the email."


def test_estimate_includes_turn_gaps():
    """Inter-turn silence stops being a rounding error on rapid exchanges."""
    narration = "one two three four five six"  # 6 words
    solo = estimate_segment_duration(narration, n_turns=0)
    dialogue = estimate_segment_duration(narration, n_turns=6, turn_gap=0.3)
    assert solo == pytest.approx(6 / 2.5)
    # 5 gaps between 6 turns
    assert dialogue == pytest.approx(6 / 2.5 + 5 * 0.3)


# --- Persistence round-trip ------------------------------------------------


def test_script_json_round_trips_turns(tmp_path: Path):
    script = Script(
        strategy_name="ubu_rex",
        title="The Debraining Machine",
        segments=[
            Segment(
                index=0,
                visual_prompt="the throne room",
                turns=[
                    Turn(speaker="PERE UBU", line="Bring in the nobles."),
                    Turn(
                        speaker="MERE UBU",
                        line="You are going too far.",
                        stage_direction="wringing her hands",
                    ),
                ],
            ),
            Segment(index=1, narration="A plain narrated beat.", visual_prompt="y"),
        ],
    )
    path = tmp_path / "ubu_s01e01.json"
    script.save_json(path)
    loaded = Script.load_json(path)

    assert len(loaded.segments) == 2
    assert [t.speaker for t in loaded.segments[0].turns] == ["PERE UBU", "MERE UBU"]
    assert loaded.segments[0].turns[1].stage_direction == "wringing her hands"
    assert loaded.segments[0].narration == "Bring in the nobles. You are going too far."
    # Single-narrator segments stay turn-free.
    assert loaded.segments[1].turns == []


def test_script_json_omits_narration_for_dialogue(tmp_path: Path):
    """The chunker can emit turns only; narration is derived on load."""
    path = tmp_path / "s.json"
    path.write_text(
        '{"segments": [{"index": 0, "turns": ['
        '{"speaker": "ubu", "line": "Shittr!"},'
        '{"speaker": "mere", "line": "Indeed."}]}]}'
    )
    loaded = Script.load_json(path)
    assert loaded.segments[0].narration == "Shittr! Indeed."


def test_script_json_rejects_empty_segment(tmp_path: Path):
    """A segment with neither narration nor turns is a hard error, not a
    silently silent video."""
    path = tmp_path / "s.json"
    path.write_text('{"segments": [{"index": 3, "visual_prompt": "x"}]}')
    with pytest.raises(ValueError, match="segment 3"):
        Script.load_json(path)


def test_db_round_trips_turns(tmp_path: Path):
    from shortform.models.video import Video
    from shortform.store.db import Database

    db = Database(tmp_path / "t.db")
    db.initialize()
    video = Video(strategy_name="ubu_rex")
    db.save_video(video)

    script = Script(
        strategy_name="ubu_rex",
        segments=[
            Segment(
                index=0,
                visual_prompt="x",
                turns=[Turn(speaker="ubu", line="Shittr!", stage_direction="bellowing")],
            )
        ],
    )
    db.save_script(script, video.id)
    loaded = db.get_script(script.id)

    assert loaded is not None
    turns = loaded.segments[0].turns
    assert len(turns) == 1
    assert isinstance(turns[0], Turn), "turns must rehydrate as Turn, not raw dict"
    assert turns[0].speaker == "ubu"
    assert turns[0].stage_direction == "bellowing"


# --- VoiceCast -------------------------------------------------------------


@dataclass
class _FakeBackend:
    """Records what it was asked to synthesize."""

    backend_name: str
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    timings: bool = False

    @property
    def name(self) -> str:
        return self.backend_name

    async def synthesize(
        self, segment: Segment, output_path: Path, config: dict[str, Any]
    ) -> TTSOutput:
        self.calls.append((segment.narration, config))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-mp3")
        words = segment.narration.split()
        timings = (
            [WordTiming(word=w, start=i * 0.5, duration=0.5) for i, w in enumerate(words)]
            if self.timings
            else []
        )
        return TTSOutput(audio_path=output_path, duration=1.0, word_timings=timings)


def _cast(**kwargs: Any) -> tuple[VoiceCast, dict[str, _FakeBackend]]:
    backends = {"f5_tts": _FakeBackend("f5_tts"), "edge": _FakeBackend("edge")}
    defaults: dict[str, Any] = {
        "base_config": {"ref_audio": "narrator.wav", "ref_text": "hello"},
        "default_backend": "f5_tts",
        "backend_factory": lambda name: backends[name],
    }
    defaults.update(kwargs)
    return VoiceCast(**defaults), backends


def test_no_cast_declared_uses_single_voice():
    """Strategies without a voices map behave exactly as before."""
    cast, _ = _cast()
    assert not cast.has_cast
    assert cast.resolve("").backend_name == "f5_tts"
    # Even a speaker-bearing script degrades to the one configured voice.
    assert cast.resolve("anyone").backend_name == "f5_tts"
    assert cast.knows("anyone")


def test_speaker_inherits_base_config_and_overrides():
    cast, _ = _cast(
        voices={"pere_ubu": {"ref_audio": "ubu.wav", "speed": 1.15}},
    )
    assignment = cast.resolve("pere_ubu")
    assert assignment.backend_name == "f5_tts"
    assert assignment.config["ref_audio"] == "ubu.wav"  # overridden
    assert assignment.config["ref_text"] == "hello"  # inherited
    assert assignment.config["speed"] == 1.15


def test_mixed_backends_within_one_cast():
    """The reason this exists: clones for principals, Edge for bit parts."""
    cast, _ = _cast(
        voices={
            "pere_ubu": {"ref_audio": "ubu.wav"},
            "bordure": {"backend": "edge", "voice": "en-GB-RyanNeural"},
        },
    )
    assert cast.resolve("pere_ubu").backend_name == "f5_tts"
    bordure = cast.resolve("bordure")
    assert bordure.backend_name == "edge"
    assert bordure.config["voice"] == "en-GB-RyanNeural"
    # `backend` is a cast-control key and must not reach the backend as config.
    assert "backend" not in bordure.config


def test_speaker_names_normalize():
    """Play texts write 'PÈRE UBU'; YAML keys want pere_ubu."""
    assert normalize_speaker("  PERE   UBU ") == "pere_ubu"
    assert normalize_speaker("Captain-Bordure") == "captain_bordure"
    cast, _ = _cast(voices={"Pere Ubu": {"ref_audio": "u.wav"}})
    assert cast.resolve("PERE UBU").config["ref_audio"] == "u.wav"


def test_uncast_speaker_raises():
    """A typo must not silently narrate a character in the wrong voice."""
    cast, _ = _cast(voices={"pere_ubu": {"ref_audio": "u.wav"}})
    assert not cast.knows("bordure")
    with pytest.raises(UnknownSpeakerError, match="bordure"):
        cast.resolve("bordure")


def test_backend_instances_are_cached():
    cast, backends = _cast(
        voices={"a": {"ref_audio": "a.wav"}, "b": {"ref_audio": "b.wav"}}
    )
    first = cast.backend_for("f5_tts")
    second = cast.backend_for("f5_tts")
    assert first is second is backends["f5_tts"]


def test_gap_after_lengthens_before_stage_direction():
    """The beat belongs before the annotated delivery, and the last turn gets
    no trailing silence."""
    cast, _ = _cast(turn_gap=0.2, stage_direction_gap=0.9)
    turns = [
        Turn(speaker="a", line="one"),
        Turn(speaker="b", line="two", stage_direction="aside"),
        Turn(speaker="a", line="three"),
    ]
    assert cast.gap_after(0, turns) == 0.9  # next line has a stage direction
    assert cast.gap_after(1, turns) == 0.2
    assert cast.gap_after(2, turns) == 0.0  # last turn


# --- TTSStage end-to-end ---------------------------------------------------


def _dialogue_ctx(tmp_path: Path, segments: list[Segment]) -> Any:
    """A PipelineContext wired to a mixed-backend Ubu cast."""
    from shortform.config import AppSettings, StrategyConfig
    from shortform.models.video import Video
    from shortform.pipeline.context import PipelineContext

    strategy = StrategyConfig(
        name="ubu_rex",
        tts={
            "backend": "f5_tts",
            "ref_audio": "narrator.wav",
            "ref_text": "hello",
            "turn_gap": 0.2,
            "stage_direction_gap": 0.8,
            "voices": {
                "pere_ubu": {"ref_audio": "ubu.wav", "speed": 1.05},
                "mere_ubu": {"ref_audio": "mere.wav"},
                "bordure": {"backend": "edge", "voice": "en-GB-RyanNeural"},
            },
        },
        visuals={"subtitles": False},  # keep Whisper out of the unit test
    )
    return PipelineContext(
        settings=AppSettings(),
        strategy=strategy,
        video=Video(strategy_name="ubu_rex"),
        script=Script(strategy_name="ubu_rex", segments=segments),
    )


@pytest.mark.asyncio
async def test_stage_synthesizes_each_turn_in_its_own_voice(tmp_path: Path, monkeypatch):
    """The payoff: one segment, three speakers, two backends, one audio file."""
    from shortform.stages import tts as tts_stage

    seg = Segment(
        index=0,
        visual_prompt="the throne room",
        turns=[
            Turn(speaker="PERE UBU", line="Bring in the nobles."),
            Turn(speaker="MERE UBU", line="You are going too far."),
            Turn(speaker="BORDURE", line="Sire, the army waits.", stage_direction="saluting"),
        ],
    )
    ctx = _dialogue_ctx(tmp_path, [seg])

    backends = {"f5_tts": _FakeBackend("f5_tts"), "edge": _FakeBackend("edge")}
    monkeypatch.setattr(tts_stage, "get_backend", lambda name, **kw: backends[name])
    monkeypatch.setattr(
        tts_stage, "concat_turn_audio", lambda inputs, output, **kw: output.write_bytes(b"joined")
    )
    monkeypatch.setattr(tts_stage, "get_audio_duration", lambda p: 3.7)
    monkeypatch.setattr(
        tts_stage.FileStore, "__init__", lambda self, base_dir=None: _init_store(self, tmp_path)
    )

    await tts_stage.TTSStage().execute(ctx)

    # Two Ubu lines went to the clone backend, Bordure's to Edge.
    f5_lines = [line for line, _ in backends["f5_tts"].calls]
    edge_lines = [line for line, _ in backends["edge"].calls]
    assert f5_lines == ["Bring in the nobles.", "You are going too far."]
    assert edge_lines == ["Sire, the army waits."]

    # Each clone got its own reference audio, inheriting ref_text from the base.
    ubu_config = backends["f5_tts"].calls[0][1]
    mere_config = backends["f5_tts"].calls[1][1]
    assert ubu_config["ref_audio"] == "ubu.wav"
    assert ubu_config["speed"] == 1.05
    assert mere_config["ref_audio"] == "mere.wav"
    assert mere_config["ref_text"] == "hello"

    # Downstream sees a single segment audio file, exactly as for one voice.
    assert seg.audio_path.endswith("segment_00.mp3")
    assert seg.actual_duration == 3.7
    assert ctx.script.total_duration == 3.7


def _init_store(store: Any, base: Path) -> None:
    """Point a FileStore at a tmp dir without touching the repo's data/."""
    store.base_dir = base
    store.videos_dir = base / "videos"
    store.assets_dir = base / "assets"
    store.videos_dir.mkdir(parents=True, exist_ok=True)
    store.assets_dir.mkdir(parents=True, exist_ok=True)


@pytest.mark.asyncio
async def test_stage_still_handles_single_narrator(tmp_path: Path, monkeypatch):
    """No regression for gothic_vignette / motivation_quotes."""
    from shortform.stages import tts as tts_stage

    seg = Segment(index=0, narration="Bartholomew read the email.", visual_prompt="x")
    ctx = _dialogue_ctx(tmp_path, [seg])
    ctx.strategy.tts.pop("voices")  # single-voice strategy

    backend = _FakeBackend("f5_tts")
    monkeypatch.setattr(tts_stage, "get_backend", lambda name, **kw: backend)
    monkeypatch.setattr(
        tts_stage.FileStore, "__init__", lambda self, base_dir=None: _init_store(self, tmp_path)
    )

    await tts_stage.TTSStage().execute(ctx)

    assert [line for line, _ in backend.calls] == ["Bartholomew read the email."]
    assert seg.actual_duration == 1.0


def test_stage_validate_rejects_uncast_speaker(tmp_path: Path):
    """Catch a typo'd speaker before burning twenty minutes of inference."""
    from shortform.stages.tts import TTSStage

    seg = Segment(
        index=0,
        visual_prompt="x",
        turns=[Turn(speaker="PERE UBU", line="Shittr!"), Turn(speaker="Lucky", line="Who?")],
    )
    errors = TTSStage().validate(_dialogue_ctx(tmp_path, [seg]))
    # Reported as written in the script, so it's greppable there; normalize_speaker
    # means either 'Lucky' or 'lucky' works once added to voices.
    assert any("Uncast speakers" in e and "Lucky" in e for e in errors)
    # The cast members that DO resolve aren't reported.
    assert not any("PERE UBU" in e for e in errors)


def test_stage_validate_accepts_full_cast(tmp_path: Path):
    from shortform.stages.tts import TTSStage

    seg = Segment(
        index=0,
        visual_prompt="x",
        turns=[Turn(speaker="PERE UBU", line="Shittr!"), Turn(speaker="Bordure", line="Sire.")],
    )
    assert TTSStage().validate(_dialogue_ctx(tmp_path, [seg])) == []


# --- Word-timing merge -----------------------------------------------------


def test_merge_offsets_timings_by_turn_position():
    merged = _merge_turn_timings(
        per_turn=[
            [WordTiming(word="one", start=0.0, duration=0.4)],
            [WordTiming(word="two", start=0.0, duration=0.4)],
        ],
        durations=[1.0, 1.0],
        gaps=[0.25, 0.0],
    )
    assert [w.word for w in merged] == ["one", "two"]
    assert merged[0].start == pytest.approx(0.0)
    # second turn starts after turn 0's duration plus the gap
    assert merged[1].start == pytest.approx(1.25)


def test_merge_is_all_or_nothing():
    """Partial timings would caption only the Edge lines and drop the cloned
    ones — worse than none, because it looks like it worked."""
    merged = _merge_turn_timings(
        per_turn=[[WordTiming(word="one", start=0.0, duration=0.4)], []],
        durations=[1.0, 1.0],
        gaps=[0.25, 0.0],
    )
    assert merged == []


# --- Concat ----------------------------------------------------------------


def _tone(path: Path, seconds: float, rate: int, channels: int) -> Path:
    """A real MP3, so the concat filter is exercised against actual ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={seconds}:sample_rate={rate}",
            "-ac", str(channels), "-c:a", "libmp3lame", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def test_concat_joins_heterogeneous_inputs_with_gaps(tmp_path: Path):
    """Mixed backends mean mismatched sample rates and channel counts — the
    concat demuxer would mangle these, which is why we use the filter."""
    a = _tone(tmp_path / "a.mp3", 1.0, 24000, 1)  # F5-shaped
    b = _tone(tmp_path / "b.mp3", 1.0, 44100, 2)  # Edge-shaped
    out = tmp_path / "joined.mp3"

    concat_turn_audio([a, b], out, gaps=[0.5, 0.0])

    assert out.exists()
    assert _duration(out) == pytest.approx(2.5, abs=0.05)


def test_concat_forces_zero_trailing_gap(tmp_path: Path):
    """A trailing pad would desync the video mux in assembly."""
    a = _tone(tmp_path / "a.mp3", 1.0, 24000, 1)
    b = _tone(tmp_path / "b.mp3", 1.0, 24000, 1)
    out = tmp_path / "joined.mp3"

    concat_turn_audio([a, b], out, gaps=[0.25, 9.0])

    assert _duration(out) == pytest.approx(2.25, abs=0.05)


def test_concat_single_input(tmp_path: Path):
    a = _tone(tmp_path / "a.mp3", 1.0, 44100, 2)
    out = tmp_path / "joined.mp3"

    concat_turn_audio([a], out)

    assert _duration(out) == pytest.approx(1.0, abs=0.05)


def test_concat_rejects_empty_and_missing(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one input"):
        concat_turn_audio([], tmp_path / "o.mp3")
    with pytest.raises(FileNotFoundError, match="missing audio"):
        concat_turn_audio([tmp_path / "nope.mp3"], tmp_path / "o.mp3")


@pytest.mark.asyncio
async def test_tts_records_turn_timings(tmp_path: Path, monkeypatch):
    """Timings must reflect the gaps too, or the schedule drifts out of sync."""
    from shortform.stages import tts as tts_stage

    seg = Segment(
        index=0,
        visual_prompt="x",
        turns=[
            Turn(speaker="PERE UBU", line="Shitre!"),
            Turn(speaker="MERE UBU", line="Indeed."),
        ],
    )
    ctx = _dialogue_ctx(tmp_path, [seg])
    ctx.strategy.tts["voices"] = {"pere_ubu": {}, "mere_ubu": {}}

    backend = _FakeBackend("f5_tts")  # every turn reports duration 1.0
    monkeypatch.setattr(tts_stage, "get_backend", lambda name, **kw: backend)
    monkeypatch.setattr(
        tts_stage, "concat_turn_audio", lambda inputs, output, **kw: output.write_bytes(b"j")
    )
    monkeypatch.setattr(tts_stage, "get_audio_duration", lambda p: 2.3)
    monkeypatch.setattr(
        tts_stage.FileStore, "__init__", lambda self, base_dir=None: _init_store(self, tmp_path)
    )

    await tts_stage.TTSStage().execute(ctx)

    assert [t.speaker for t in seg.turn_timings] == ["PERE UBU", "MERE UBU"]
    assert seg.turn_timings[0].start == pytest.approx(0.0)
    # second turn starts after turn 0's duration PLUS the configured turn_gap
    assert seg.turn_timings[1].start == pytest.approx(1.0 + ctx.strategy.tts["turn_gap"])
