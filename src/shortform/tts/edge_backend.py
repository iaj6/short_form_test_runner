"""Edge TTS backend — Microsoft Edge's free streaming TTS with sentence boundary events.

Provides word-level timings via SentenceBoundary events, which downstream
assembly uses for the animated phrase-level subtitles.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import edge_tts

from shortform.models.script import Segment, WordTiming
from shortform.tts.backend import TTSOutput, get_audio_duration

logger = logging.getLogger(__name__)


class EdgeBackend:
    """Microsoft Edge TTS — free, streaming, no API key required."""

    @property
    def name(self) -> str:
        return "edge"

    async def synthesize(
        self,
        segment: Segment,
        output_path: Path,
        config: dict[str, Any],
    ) -> TTSOutput:
        voice = config.get("voice", "en-US-AriaNeural")
        rate = config.get("rate", "+5%")
        volume = config.get("volume", "+0%")

        logger.info("Edge TTS segment %d: voice=%s rate=%s", segment.index, voice, rate)

        communicate = edge_tts.Communicate(
            text=segment.narration,
            voice=voice,
            rate=rate,
            volume=volume,
        )

        sentence_boundaries: list[dict[str, object]] = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "SentenceBoundary":
                    sentence_boundaries.append(chunk)

        word_timings = _sentence_boundaries_to_word_timings(sentence_boundaries)
        duration = get_audio_duration(output_path)

        return TTSOutput(
            audio_path=output_path,
            duration=duration,
            word_timings=word_timings,
        )


def _sentence_boundaries_to_word_timings(
    boundaries: list[dict[str, object]],
) -> list[WordTiming]:
    """Split sentence boundaries into approximate per-word timings.

    Edge TTS v7 only emits SentenceBoundary events. We distribute the words
    of each sentence evenly across the sentence's time window.
    """
    timings: list[WordTiming] = []
    for boundary in boundaries:
        text = str(boundary["text"])
        offset = float(boundary["offset"]) / 1e7  # 100ns ticks → seconds  # type: ignore[arg-type]
        duration = float(boundary["duration"]) / 1e7  # type: ignore[arg-type]
        words = text.split()
        if not words:
            continue
        word_duration = duration / len(words)
        for i, word in enumerate(words):
            timings.append(WordTiming(
                word=word,
                start=offset + i * word_duration,
                duration=word_duration,
            ))
    return timings
