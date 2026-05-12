"""TTS stage — dispatches to a pluggable backend (Edge, F5-TTS, ...)."""

from __future__ import annotations

import logging
from typing import Any

from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.tts.registry import get_backend

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
        backend_name, config = _resolve_backend_config(ctx)
        backend = get_backend(backend_name, **_backend_init_kwargs(backend_name, ctx))

        total_duration = 0.0
        for seg in ctx.script.segments:
            output_path = file_store.segment_audio_path(ctx.video.id, seg.index)
            logger.info(
                "TTS segment %d via %s: %s...",
                seg.index, backend.name, seg.narration[:50],
            )
            result = await backend.synthesize(
                segment=seg,
                output_path=output_path,
                config=config,
            )
            seg.audio_path = str(result.audio_path)
            seg.actual_duration = result.duration
            seg.word_timings = result.word_timings
            total_duration += result.duration

            logger.info(
                "Segment %d: %.1fs (estimated %.1fs)",
                seg.index, result.duration, seg.estimated_duration,
            )

        ctx.script.total_duration = total_duration
        ctx.video.duration = total_duration
        ctx.video.status = VideoStatus.TTS_DONE
        logger.info("TTS complete: %.1fs total audio via %s", total_duration, backend.name)
        return ctx


def _resolve_backend_config(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Merge defaults (settings.tts) with strategy overrides (strategy.tts +
    legacy strategy.content voice/rate) into a single config dict.

    Strategy.tts wins over strategy.content wins over settings.tts. The legacy
    strategy.content.{voice,rate} path is kept so existing strategy YAMLs
    (motivation_quotes, tech_tips) don't need to migrate.
    """
    settings_tts = ctx.settings.tts
    config: dict[str, Any] = {
        "voice": settings_tts.voice,
        "rate": settings_tts.rate,
        "volume": settings_tts.volume,
    }
    if "voice" in ctx.strategy.content:
        config["voice"] = ctx.strategy.content["voice"]
    if "rate" in ctx.strategy.content:
        config["rate"] = ctx.strategy.content["rate"]
    config.update(ctx.strategy.tts)

    backend_name = ctx.strategy.tts.get("backend") or getattr(settings_tts, "backend", "edge")
    return backend_name, config


def _backend_init_kwargs(backend_name: str, ctx: PipelineContext) -> dict[str, Any]:
    """Backend-specific constructor kwargs from settings."""
    if backend_name == "f5_tts":
        cli_path = getattr(ctx.settings.tts, "f5_tts_cli", None)
        if cli_path:
            return {"cli_path": cli_path}
    return {}
