"""TTS stage — Edge TTS produces audio per segment and measures durations."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import edge_tts

from shortform.models.script import WordTiming
from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore

logger = logging.getLogger(__name__)


class TTSStage:
    @property
    def name(self) -> str:
        return "tts"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.script.segments:
            errors.append("No script segments to synthesize")
        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        file_store = FileStore()
        tts_cfg = ctx.settings.tts
        total_duration = 0.0

        for seg in ctx.script.segments:
            output_path = file_store.segment_audio_path(ctx.video.id, seg.index)
            logger.info("Generating TTS for segment %d: %s...", seg.index, seg.narration[:50])

            # Strategy can override voice and rate
            voice = ctx.strategy.content.get("voice", tts_cfg.voice)
            rate = ctx.strategy.content.get("rate", tts_cfg.rate)

            communicate = edge_tts.Communicate(
                text=seg.narration,
                voice=voice,
                rate=rate,
                volume=tts_cfg.volume,
            )

            # Stream to capture sentence boundary timings alongside the audio
            sentence_boundaries: list[dict[str, object]] = []
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == "SentenceBoundary":
                        sentence_boundaries.append(chunk)

            # Derive word-level timings by splitting sentences evenly
            seg.word_timings = _sentence_boundaries_to_word_timings(
                sentence_boundaries
            )

            # Measure actual audio duration
            duration = _get_audio_duration(output_path)
            seg.actual_duration = duration
            seg.audio_path = str(output_path)
            total_duration += duration

            logger.info(
                "Segment %d: %.1fs (estimated %.1fs)",
                seg.index,
                duration,
                seg.estimated_duration,
            )

        ctx.script.total_duration = total_duration
        ctx.video.duration = total_duration
        ctx.video.status = VideoStatus.TTS_DONE

        logger.info("TTS complete: %.1fs total audio", total_duration)
        return ctx


def _sentence_boundaries_to_word_timings(
    boundaries: list[dict[str, object]],
) -> list[WordTiming]:
    """Split sentence boundaries into approximate per-word timings.

    Edge TTS v7 only provides SentenceBoundary events. We distribute
    words evenly within each sentence's time window.
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


def _get_audio_duration(path: Path) -> float:
    """Get audio duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        # Fallback: rough estimate from file size
        size = path.stat().st_size
        return size / 16000 if size > 0 else 0.0
