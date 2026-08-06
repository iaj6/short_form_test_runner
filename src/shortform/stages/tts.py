"""TTS stage — dispatches to a pluggable backend (Edge, F5-TTS, ...).

Two shapes of segment:

- **Single narrator** (motivation_quotes, gothic_vignette) — one voice for the
  whole video, synthesized straight from `seg.narration`.
- **Dialogue** (adapted plays, multi-character sketches) — `seg.turns` carries
  per-speaker lines, each synthesized in that speaker's voice (possibly a
  different backend per speaker) and concatenated into one segment track.

Both produce exactly one audio file at `seg.audio_path`, which is all assembly
ever reads — so multi-speaker support needs no changes downstream of here.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from shortform.models.script import Segment, Turn, TurnTiming, WordTiming
from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.tts.backend import TTSOutput, get_audio_duration
from shortform.tts.cast import (
    CAST_CONTROL_KEYS,
    DEFAULT_STAGE_DIRECTION_GAP,
    DEFAULT_TURN_GAP,
    VoiceCast,
)
from shortform.tts.concat import concat_turn_audio
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

        for seg in ctx.script.segments:
            if seg.turns:
                if not any(t.line.strip() for t in seg.turns):
                    errors.append(f"Segment {seg.index}: all dialogue turns are empty")
            elif not seg.narration.strip():
                errors.append(f"Segment {seg.index}: no narration and no turns")

        # Resolve the whole cast up front. An uncast speaker is fatal, and
        # finding out here beats finding out twenty minutes into a batch run
        # after paying for the segments that happened to come first.
        try:
            cast = _build_voice_cast(ctx)
        except Exception as e:  # noqa: BLE001 - surfaced as a validation error
            errors.append(f"TTS voice cast config is invalid: {e}")
        else:
            uncast = sorted(
                {
                    t.speaker
                    for seg in ctx.script.segments
                    for t in seg.turns
                    if t.line.strip() and not cast.knows(t.speaker)
                }
            )
            if uncast:
                errors.append(
                    f"Uncast speakers: {', '.join(uncast)}. "
                    f"Add them to strategy.tts.voices (cast: {', '.join(cast.speakers)})"
                )
        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        file_store = FileStore()
        cast = _build_voice_cast(ctx)
        logger.info("TTS voice cast: %s", cast.summary())

        total_duration = 0.0
        for seg in ctx.script.segments:
            output_path = file_store.segment_audio_path(ctx.video.id, seg.index)

            if seg.is_dialogue:
                result = await _synthesize_dialogue(
                    seg, output_path, cast, file_store, ctx
                )
            else:
                result = await _synthesize_single(seg, output_path, cast)

            seg.audio_path = str(result.audio_path)
            seg.actual_duration = result.duration
            seg.word_timings = result.word_timings
            # Backends that don't emit word timings (F5-TTS) get them recovered
            # via Whisper so the animated-caption path lights up — but only when
            # the strategy opts into subtitles. Soft-deps on faster-whisper;
            # degrades to no captions if it isn't installed.
            if not seg.word_timings and _want_captions(ctx):
                from shortform.tts.whisper_align import DEFAULT_MODEL, align_words

                model_size = ctx.strategy.visuals.get("caption_model", DEFAULT_MODEL)
                seg.word_timings = align_words(Path(seg.audio_path), model_size=model_size)
                if seg.word_timings:
                    logger.info(
                        "Recovered %d caption words for segment %d via Whisper",
                        len(seg.word_timings), seg.index,
                    )
            total_duration += result.duration

            logger.info(
                "Segment %d: %.1fs (estimated %.1fs)",
                seg.index, result.duration, seg.estimated_duration,
            )

        ctx.script.total_duration = total_duration
        ctx.video.duration = total_duration
        ctx.video.status = VideoStatus.TTS_DONE
        logger.info("TTS complete: %.1fs total audio", total_duration)
        return ctx


async def _synthesize_single(
    seg: Segment, output_path: Path, cast: VoiceCast
) -> TTSOutput:
    """One voice, straight from the segment's narration (the original path)."""
    assignment = cast.resolve("")
    backend = cast.backend_for(assignment.backend_name)
    logger.info(
        "TTS segment %d via %s: %s...", seg.index, backend.name, seg.narration[:50]
    )
    return await backend.synthesize(
        segment=seg, output_path=output_path, config=assignment.config
    )


async def _synthesize_dialogue(
    seg: Segment,
    output_path: Path,
    cast: VoiceCast,
    file_store: FileStore,
    ctx: PipelineContext,
) -> TTSOutput:
    """A voice per turn, concatenated into one segment track.

    Per-turn files are kept alongside the other working assets rather than in a
    temp dir — a run that produces an odd-sounding exchange is much easier to
    debug when you can play the individual lines.
    """
    turns = [t for t in seg.turns if t.line.strip()]
    if not turns:
        raise RuntimeError(
            f"Segment {seg.index} is marked as dialogue but every turn is empty"
        )

    turn_dir = file_store.video_dir(ctx.video.id) / f"segment_{seg.index:02d}_turns"
    turn_dir.mkdir(parents=True, exist_ok=True)

    turn_paths: list[Path] = []
    gaps: list[float] = []
    durations: list[float] = []
    per_turn_timings: list[list[WordTiming]] = []

    for t_idx, turn in enumerate(turns):
        assignment = cast.resolve(turn.speaker)
        backend = cast.backend_for(assignment.backend_name)
        turn_path = turn_dir / f"turn_{t_idx:02d}.mp3"

        # A per-turn view of the segment: backends read `narration`, so hand them
        # this turn's line. turns=[] prevents the derived-narration path in
        # __post_init__ from overwriting it.
        turn_segment = replace(
            seg,
            narration=turn.line.strip(),
            turns=[],
            word_timings=[],
            audio_path="",
            actual_duration=0.0,
        )

        logger.info(
            "TTS segment %d turn %d/%d [%s via %s]: %s...",
            seg.index, t_idx + 1, len(turns), assignment.speaker or "default",
            backend.name, turn.line[:40],
        )
        result = await backend.synthesize(
            segment=turn_segment, output_path=turn_path, config=assignment.config
        )
        turn_paths.append(Path(result.audio_path))
        durations.append(result.duration)
        per_turn_timings.append(result.word_timings)
        gaps.append(cast.gap_after(t_idx, turns))

    concat_turn_audio(turn_paths, output_path, gaps=gaps)

    # Record where each turn landed in the joined audio. VisualGenStage slices
    # this per Veo clip so the video model is told who is speaking when —
    # otherwise it animates every puppet's mouth simultaneously, which is the
    # single most obviously wrong thing about generated dialogue.
    seg.turn_timings = []
    offset = 0.0
    for turn, turn_duration, gap in zip(turns, durations, gaps, strict=True):
        seg.turn_timings.append(
            TurnTiming(speaker=turn.speaker, start=offset, duration=turn_duration)
        )
        offset += turn_duration + gap

    # Measure the joined file rather than summing the parts: the concat filter's
    # resample + MP3 frame padding shifts the total by a few milliseconds, and
    # assembly muxes video against this number.
    duration = get_audio_duration(output_path)
    logger.info(
        "Segment %d: joined %d turns (%s) → %.1fs",
        seg.index, len(turns),
        "/".join(_speaker_labels(turns)),
        duration,
    )
    return TTSOutput(
        audio_path=output_path,
        duration=duration,
        word_timings=_merge_turn_timings(per_turn_timings, durations, gaps),
    )


def _merge_turn_timings(
    per_turn: list[list[WordTiming]],
    durations: list[float],
    gaps: list[float],
) -> list[WordTiming]:
    """Shift each turn's word timings into concatenated-segment time.

    All-or-nothing: if ANY turn's backend emitted no timings (F5-TTS never
    does), return nothing so the caller falls back to Whisper over the whole
    segment. Merging partial timings would caption only the Edge-voiced lines
    and silently drop the cloned ones — worse than no captions, because it
    looks like it worked.
    """
    if not per_turn or any(not timings for timings in per_turn):
        return []

    merged: list[WordTiming] = []
    offset = 0.0
    for timings, duration, gap in zip(per_turn, durations, gaps, strict=True):
        for w in timings:
            merged.append(
                WordTiming(word=w.word, start=w.start + offset, duration=w.duration)
            )
        offset += duration + gap
    return merged


def _speaker_labels(turns: list[Turn]) -> list[str]:
    """Ordered distinct speakers, for a compact log line."""
    seen: list[str] = []
    for t in turns:
        label = t.speaker or "default"
        if label not in seen:
            seen.append(label)
    return seen


def _build_voice_cast(ctx: PipelineContext) -> VoiceCast:
    """Assemble the run's VoiceCast from settings + strategy config."""
    backend_name, config = _resolve_backend_config(ctx)
    tts_cfg = ctx.strategy.tts

    return VoiceCast(
        base_config={k: v for k, v in config.items() if k not in CAST_CONTROL_KEYS},
        default_backend=backend_name,
        voices=tts_cfg.get("voices") or {},
        backend_factory=lambda name: get_backend(
            name, **_backend_init_kwargs(name, ctx)
        ),
        turn_gap=float(tts_cfg.get("turn_gap", DEFAULT_TURN_GAP)),
        stage_direction_gap=float(
            tts_cfg.get("stage_direction_gap", DEFAULT_STAGE_DIRECTION_GAP)
        ),
    )


def _resolve_backend_config(ctx: PipelineContext) -> tuple[str, dict[str, Any]]:
    """Merge defaults (settings.tts) with strategy overrides (strategy.tts +
    legacy strategy.content voice/rate) into a single config dict.

    Strategy.tts wins over strategy.content wins over settings.tts. The legacy
    strategy.content.{voice,rate} path is kept so existing strategy YAMLs
    (motivation_quotes, tech_tips) don't need to migrate.

    This defines the DEFAULT voice. Per-speaker overrides layer on top of it in
    VoiceCast; cast-control keys (voices, turn_gap, ...) are stripped there
    before the remainder reaches a backend.
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

    backend_name = str(
        ctx.strategy.tts.get("backend") or getattr(settings_tts, "backend", "edge")
    )
    return backend_name, config


def _want_captions(ctx: PipelineContext) -> bool:
    """Whether to recover word timings for animated captions when the backend
    didn't provide them. Strategy opt-in via visuals.subtitles (default True)."""
    return bool(ctx.strategy.visuals.get("subtitles", True))


def _backend_init_kwargs(backend_name: str, ctx: PipelineContext) -> dict[str, Any]:
    """Backend-specific constructor kwargs from settings."""
    if backend_name == "f5_tts":
        cli_path = getattr(ctx.settings.tts, "f5_tts_cli", None)
        if cli_path:
            return {"cli_path": cli_path}
    if backend_name == "elevenlabs":
        import os

        key = ctx.settings.elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        return {"api_key": key}
    return {}
