"""Whisper word-timing alignment.

Recovers per-word timestamps for TTS backends that don't emit them — notably
F5-TTS — so the animated-subtitle (karaoke caption) path in assembly works for
voice-cloned strategies, not just Edge TTS.

Uses faster-whisper (CTranslate2, no torch) as a SOFT dependency: if it isn't
installed, alignment is skipped and the caller falls back to no captions rather
than crashing. This keeps the project venv slim by default; the model weights
download on first use.

    uv pip install faster-whisper

Note: this is free transcription, not forced alignment against the known
script text, so caption words come from ASR and may occasionally differ from
the narration. For short, clearly-spoken narration this is reliable enough;
upgrade to a forced-alignment lib (whisperx / stable-ts) if drift matters.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from shortform.models.script import WordTiming

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base"


@lru_cache(maxsize=2)
def _load_model(model_size: str) -> Any:
    """Load (and cache) a faster-whisper model. Raises ImportError if the
    package isn't installed — callers treat that as 'captions unavailable'."""
    from faster_whisper import WhisperModel  # soft import — may be absent

    logger.info(
        "Loading faster-whisper model '%s' (first run downloads weights)", model_size
    )
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def is_available() -> bool:
    """True if faster-whisper can be imported in this environment."""
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False


def align_words(
    audio_path: Path,
    model_size: str = DEFAULT_MODEL,
    language: str = "en",
) -> list[WordTiming]:
    """Transcribe `audio_path` and return per-word timings.

    Returns an empty list (and logs a warning) if faster-whisper isn't
    installed, so the caller degrades gracefully to "no captions".
    """
    try:
        model = _load_model(model_size)
    except ImportError:
        logger.warning(
            "faster-whisper not installed; skipping caption alignment for %s. "
            "Install with: uv pip install faster-whisper",
            audio_path.name,
        )
        return []

    segments, _info = model.transcribe(
        str(audio_path), language=language, word_timestamps=True
    )

    timings: list[WordTiming] = []
    for seg in segments:
        for w in seg.words or []:
            word = w.word.strip()
            if not word:
                continue
            start = float(w.start)
            end = float(w.end)
            timings.append(
                WordTiming(word=word, start=start, duration=max(0.0, end - start))
            )
    return timings
