"""Tests for the assembly stage's pure helpers and failure guards.

assembly.py is the terminal stage every video flows through and carries the
most hand-built ffmpeg filtergraph logic, yet its failures tend to be *silent*
(a wrong xfade offset is a glitch, not a crash). These tests protect the cheap,
high-leverage pure helpers plus the two failure guards added alongside them:
  - _probe_duration now raises instead of returning a poisoning 0.0
  - _mix_background_music tolerates music_volume == 0 (no ZeroDivisionError)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import ImageFont

from shortform.models.script import Script, Segment, WordTiming
from shortform.models.video import Video
from shortform.pipeline.context import PipelineContext
from shortform.stages import assembly
from shortform.stages.assembly import (
    AssemblyStage,
    _group_words_into_phrases,
    _mix_background_music,
    _mux_video_with_audio,
    _probe_duration,
    _wrap_text_to_width,
)
from shortform.visuals.backend import VisualOutputType

# --- _group_words_into_phrases -------------------------------------------------


def test_group_words_chunks_by_max_words():
    words = [WordTiming(word=f"w{i}", start=float(i), duration=1.0) for i in range(7)]
    phrases = _group_words_into_phrases(words, max_words=3)
    # 7 words / 3 per chunk → 3 phrases (3, 3, 1)
    assert [p[0] for p in phrases] == ["w0 w1 w2", "w3 w4 w5", "w6"]


def test_group_words_phrase_end_is_next_phrase_start():
    words = [WordTiming(word=f"w{i}", start=float(i), duration=1.0) for i in range(6)]
    phrases = _group_words_into_phrases(words, max_words=3)
    # First phrase starts at w0 (0.0) and ends at the start of w3 (3.0).
    assert phrases[0][1] == 0.0
    assert phrases[0][2] == 3.0


def test_group_words_last_phrase_end_uses_last_word_duration():
    words = [WordTiming(word=f"w{i}", start=float(i), duration=1.0) for i in range(5)]
    phrases = _group_words_into_phrases(words, max_words=3)
    # Last chunk is [w3, w4]; end = last.start (4.0) + last.duration (1.0) = 5.0
    assert phrases[-1][2] == 5.0


def test_group_words_empty_input():
    assert _group_words_into_phrases([], max_words=3) == []


# --- _wrap_text_to_width -------------------------------------------------------


def _font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in assembly._FONT_PATHS:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, 40)
            except Exception:
                continue
    return ImageFont.load_default()


def test_wrap_text_wraps_when_too_wide():
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text = "the quick brown fox jumps over the lazy dog several more words here"
    lines = _wrap_text_to_width(draw, text, _font(), max_width=200)
    # Should produce more than one line and lose no words.
    assert len(lines) > 1
    assert " ".join(lines).split() == text.split()


def test_wrap_text_oversized_single_word_still_emitted():
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    # A single word wider than max_width must not be dropped (the `or not
    # current_line` escape hatch keeps it).
    lines = _wrap_text_to_width(draw, "supercalifragilistic", _font(), max_width=5)
    assert lines == ["supercalifragilistic"]


# --- _probe_duration (the #3 guard) -------------------------------------------


def test_probe_duration_raises_on_missing_file(tmp_path: Path):
    # ffprobe exits non-zero on a path that doesn't exist → must raise, not 0.0.
    with pytest.raises(RuntimeError, match="ffprobe failed"):
        _probe_duration(tmp_path / "nope.mp4")


def test_probe_duration_raises_on_unparseable_output(tmp_path: Path):
    bogus = tmp_path / "not_media.txt"
    bogus.write_text("this is not a media file")
    with pytest.raises(RuntimeError):
        _probe_duration(bogus)


# --- _mix_background_music (the #7 guard) -------------------------------------


def test_mix_background_music_volume_zero_no_crash(tmp_path: Path):
    """music_volume == 0 (a valid 'mute the bed' config) must not raise a
    ZeroDivisionError; the amix weight should resolve to 0.00."""
    captured: dict[str, list[str]] = {}

    def fake_run_ffmpeg(args: list[str]) -> None:
        captured["args"] = args

    with patch.object(assembly, "_probe_duration", return_value=12.0), \
         patch.object(assembly, "_run_ffmpeg", side_effect=fake_run_ffmpeg):
        _mix_background_music(
            video_path=tmp_path / "v.mp4",
            music_path=tmp_path / "m.mp3",
            output_path=tmp_path / "out.mp4",
            music_volume=0.0,
            duck_volume=0.05,
            fade_in=1.0,
            fade_out=1.0,
            audio_bitrate="192k",
            audio_sample_rate=44100,
        )

    filter_complex = captured["args"][captured["args"].index("-filter_complex") + 1]
    assert "weights=1 0.00" in filter_complex


def test_mix_background_music_weight_is_duck_over_volume(tmp_path: Path):
    captured: dict[str, list[str]] = {}

    with patch.object(assembly, "_probe_duration", return_value=10.0), \
         patch.object(assembly, "_run_ffmpeg", side_effect=lambda a: captured.update(args=a)):
        _mix_background_music(
            video_path=tmp_path / "v.mp4",
            music_path=tmp_path / "m.mp3",
            output_path=tmp_path / "out.mp4",
            music_volume=0.20,
            duck_volume=0.05,
            fade_in=1.0,
            fade_out=1.0,
            audio_bitrate="192k",
            audio_sample_rate=44100,
        )

    filter_complex = captured["args"][captured["args"].index("-filter_complex") + 1]
    # 0.05 / 0.20 = 0.25
    assert "weights=1 0.25" in filter_complex


# --- xfade offset math (multi-clip concat) ------------------------------------


def test_concat_xfade_offsets_are_monotonic_and_correct(tmp_path: Path):
    """The cumulative xfade offset for clip i is sum(durations[:i]) - i*cf.
    Regression guard for the offset arithmetic in _concat_video_clips_with_xfade.
    """
    captured: dict[str, list[str]] = {}
    durations = [8.0, 8.0, 5.0]
    cf = 0.3

    clips = []
    for i in range(len(durations)):
        p = tmp_path / f"clip_{i}.mp4"
        p.write_bytes(b"x")
        clips.append(p)

    with patch.object(assembly, "_probe_duration", side_effect=durations), \
         patch.object(assembly, "_run_ffmpeg", side_effect=lambda a: captured.update(args=a)):
        assembly._concat_video_clips_with_xfade(
            clips=clips,
            output_path=tmp_path / "out.mp4",
            crossfade_duration=cf,
            fps=30,
            width=1080,
            height=1920,
            video_bitrate="8M",
            pixel_format="yuv420p",
            preset="medium",
        )

    fc = captured["args"][captured["args"].index("-filter_complex") + 1]
    # offset for clip 1 = 8.0 - 0.3 = 7.7; for clip 2 = (8.0-0.3)+(8.0-0.3) = 15.4
    assert "offset=7.700" in fc
    assert "offset=15.400" in fc


# --- _mux_video_with_audio: short video must pad, not truncate (#4) -----------


def _mux_filter(video_dur: float, audio_dur: float, tmp_path: Path) -> str:
    captured: dict[str, list[str]] = {}
    with patch.object(assembly, "_probe_duration", side_effect=[video_dur, audio_dur]), \
         patch.object(assembly, "_run_ffmpeg", side_effect=lambda a: captured.update(args=a)):
        _mux_video_with_audio(
            video_path=tmp_path / "v.mp4",
            audio_path=tmp_path / "a.mp3",
            output_path=tmp_path / "out.mp4",
            video_bitrate="8M",
            pixel_format="yuv420p",
            preset="medium",
        )
    return captured["args"][captured["args"].index("-filter_complex") + 1]


def test_mux_short_video_holds_last_frame(tmp_path: Path):
    # Video 8s vs narration 16s — pad the video by 8s, do not truncate audio.
    fc = _mux_filter(8.0, 16.0, tmp_path)
    assert "tpad=stop_mode=clone:stop_duration=8.000" in fc
    # Audio fade anchors to the full narration length (16 - 0.05).
    assert "st=15.950" in fc


def test_mux_long_video_copies_and_lets_shortest_trim(tmp_path: Path):
    # Video 20s vs narration 16s — no padding; -shortest trims video to audio.
    fc = _mux_filter(20.0, 16.0, tmp_path)
    assert "[0:v]copy[vout]" in fc
    assert "tpad" not in fc


# --- AssemblyStage.validate: multi-clip sub-clip existence (#8) ---------------


def _ctx_with_video_segment(audio: Path, image: Path, sub_clips: list[str]) -> PipelineContext:
    seg = Segment(index=0, narration="hi", visual_prompt="x", text_overlay="")
    seg.audio_path = str(audio)
    seg.image_path = str(image)
    script = Script(segments=[seg])
    ctx = PipelineContext(video=Video(), script=script)
    ctx.artifacts["segment_visual_types"] = {0: VisualOutputType.VIDEO}
    ctx.artifacts["segment_clips"] = {0: sub_clips}
    return ctx


def test_validate_flags_missing_sub_clip(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"audio")
    clip0 = tmp_path / "clip0.mp4"
    clip0.write_bytes(b"clip0")
    # clip1 referenced but never written → must be flagged.
    missing = tmp_path / "clip1.mp4"
    ctx = _ctx_with_video_segment(audio, clip0, [str(clip0), str(missing)])

    errors = AssemblyStage().validate(ctx)
    assert any("sub-clip 1" in e for e in errors)


def test_validate_flags_empty_sub_clip(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"audio")
    clip0 = tmp_path / "clip0.mp4"
    clip0.write_bytes(b"clip0")
    empty = tmp_path / "clip1.mp4"
    empty.write_bytes(b"")  # zero-length
    ctx = _ctx_with_video_segment(audio, clip0, [str(clip0), str(empty)])

    errors = AssemblyStage().validate(ctx)
    assert any("sub-clip 1" in e for e in errors)


def test_validate_passes_when_all_sub_clips_present(tmp_path: Path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"audio")
    clip0 = tmp_path / "clip0.mp4"
    clip0.write_bytes(b"clip0")
    clip1 = tmp_path / "clip1.mp4"
    clip1.write_bytes(b"clip1")
    ctx = _ctx_with_video_segment(audio, clip0, [str(clip0), str(clip1)])

    errors = AssemblyStage().validate(ctx)
    # No sub-clip error (ffmpeg-availability error may still appear in CI).
    assert not any("sub-clip" in e for e in errors)
