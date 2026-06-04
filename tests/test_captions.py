"""Tests for Whisper caption alignment (#1) and the strategy opt-in."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shortform.config import StrategyConfig
from shortform.pipeline.context import PipelineContext
from shortform.stages.tts import _want_captions
from shortform.tts import whisper_align


def test_want_captions_default_true():
    ctx = PipelineContext(strategy=StrategyConfig(name="s", visuals={}))
    assert _want_captions(ctx) is True


def test_want_captions_opt_out():
    ctx = PipelineContext(strategy=StrategyConfig(name="s", visuals={"subtitles": False}))
    assert _want_captions(ctx) is False


def test_align_words_degrades_when_whisper_missing(tmp_path: Path):
    # Simulate faster-whisper not installed: _load_model raises ImportError.
    whisper_align._load_model.cache_clear()
    with patch.object(whisper_align, "_load_model", side_effect=ImportError("no module")):
        result = whisper_align.align_words(tmp_path / "seg.mp3")
    assert result == []


def test_align_words_parses_word_timestamps(tmp_path: Path):
    # Fake faster-whisper model: transcribe() yields segments with .words.
    fake_words = [
        SimpleNamespace(word=" Bartholomew", start=0.0, end=0.6),
        SimpleNamespace(word=" sighed", start=0.6, end=1.1),
        SimpleNamespace(word="  ", start=1.1, end=1.2),  # whitespace-only → dropped
    ]
    fake_segment = SimpleNamespace(words=fake_words)

    class FakeModel:
        def transcribe(self, *args, **kwargs):
            assert kwargs.get("word_timestamps") is True
            return [fake_segment], SimpleNamespace()

    whisper_align._load_model.cache_clear()
    with patch.object(whisper_align, "_load_model", return_value=FakeModel()):
        result = whisper_align.align_words(tmp_path / "seg.mp3")

    assert [w.word for w in result] == ["Bartholomew", "sighed"]
    assert result[0].start == 0.0
    assert abs(result[0].duration - 0.6) < 1e-6
    assert abs(result[1].duration - 0.5) < 1e-6
