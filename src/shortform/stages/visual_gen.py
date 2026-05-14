"""Visual generation stage — delegates to pluggable backends.

For video-output backends (Veo), we generate N clips per segment so the
visual track is at least as long as the F5-TTS narration. Veo is hard-locked
at ~8s per clip, so a 20s narration needs 3 clips. Still-image backends
(Pillow) keep producing one asset per segment — Ken Burns in assembly
extends to any duration.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path

from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.visuals.backend import VisualBackend, VisualOutputType
from shortform.visuals.pillow_backend import PillowBackend

logger = logging.getLogger(__name__)

# Approximate usable seconds per Veo clip when stitched with a small xfade.
# Veo 3 produces ~8s clips; we leave a bit of headroom for the inter-clip
# crossfade in assembly, so each clip "contributes" ~7.5s of timeline.
CLIP_TARGET_SECONDS = 7.5


class VisualGenStage:
    def __init__(self, backend: VisualBackend | None = None) -> None:
        self._backend = backend or PillowBackend()

    @property
    def name(self) -> str:
        return "visual_gen"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.script.segments:
            errors.append("No script segments for visual generation")
        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        file_store = FileStore()
        vis_cfg = ctx.settings.visuals
        vid_cfg = ctx.settings.video

        # Merge default visuals config with strategy overrides
        config: dict[str, object] = {
            "font_size": vis_cfg.font_size,
            "font_color": vis_cfg.font_color,
            "text_margin": vis_cfg.text_margin,
            "gradient_top": vis_cfg.gradient_top,
            "gradient_bottom": vis_cfg.gradient_bottom,
        }
        config.update(ctx.strategy.visuals)

        has_video_clips = False
        # Per-segment clip lists for assembly. Only populated when a video-output
        # segment needed >1 clip; assembly falls back to seg.image_path otherwise.
        segment_clips: dict[int, list[str]] = ctx.artifacts.setdefault(
            "segment_clips", {}
        )

        for seg in ctx.script.segments:
            video_dir = file_store.video_dir(ctx.video.id)
            # Always generate clip 0 first so we can see the output type before
            # deciding whether to generate more.
            first_output = video_dir / f"segment_{seg.index:02d}"
            logger.info(
                "Generating visual for segment %d [%s] (clip 1)",
                seg.index, self._backend.name,
            )
            first_result = await self._backend.generate(
                segment=seg,
                output_path=first_output,
                width=vid_cfg.width,
                height=vid_cfg.height,
                config=config,
            )
            seg.image_path = str(first_result.path)
            segment_types = ctx.artifacts.setdefault("segment_visual_types", {})
            segment_types[seg.index] = first_result.output_type

            if first_result.output_type == VisualOutputType.VIDEO:
                has_video_clips = True

            # Multi-clip path: if the backend produces video AND the audio is
            # longer than one Veo clip, generate additional clips to cover it.
            # Within a segment we CHAIN clips by extracting the last frame of
            # clip M and passing it as the starting image for clip M+1 (via
            # `chain_from` in config). This makes sub-clip cuts within a
            # segment continuous — Bartholomew's pose, lighting, and motion
            # pick up exactly where they left off. The first clip of each
            # segment re-anchors to the hero reference image so the character
            # doesn't drift across segments.
            clip_paths: list[str] = [str(first_result.path)]
            if first_result.output_type == VisualOutputType.VIDEO:
                target_seconds = seg.actual_duration or seg.estimated_duration
                n_clips_total = max(
                    1, math.ceil(target_seconds / CLIP_TARGET_SECONDS)
                )
                if n_clips_total > 1:
                    logger.info(
                        "Segment %d needs %d clips for %.1fs audio (chained)",
                        seg.index, n_clips_total, target_seconds,
                    )
                for extra_idx in range(1, n_clips_total):
                    # Extract last frame of the previous clip, pass it as the
                    # chain anchor for this clip.
                    prev_clip = Path(clip_paths[-1])
                    last_frame = video_dir / (
                        f"segment_{seg.index:02d}_clip_{extra_idx - 1:02d}_lastframe.png"
                        if extra_idx > 1
                        else f"segment_{seg.index:02d}_lastframe.png"
                    )
                    _extract_last_frame(prev_clip, last_frame)
                    chain_config = {**config, "chain_from": str(last_frame)}

                    extra_output = (
                        video_dir / f"segment_{seg.index:02d}_clip_{extra_idx:02d}"
                    )
                    logger.info(
                        "Generating visual for segment %d [%s] (clip %d/%d, chained)",
                        seg.index, self._backend.name, extra_idx + 1, n_clips_total,
                    )
                    extra_result = await self._backend.generate(
                        segment=seg,
                        output_path=extra_output,
                        width=vid_cfg.width,
                        height=vid_cfg.height,
                        config=chain_config,
                    )

                    # If the chained generation got rejected (e.g., Veo safety
                    # filter on the chained frame) and the backend fell back
                    # to a still image, retry once anchored to the hero ref
                    # instead — chained frames are sometimes darker/more
                    # skeletal in ways that trigger filters the clean hero
                    # doesn't. If that *also* fails, we stop multi-clip
                    # generation for this segment rather than mixing video
                    # and still-image paths through the rest of the pipeline.
                    if extra_result.output_type != VisualOutputType.VIDEO:
                        logger.warning(
                            "Segment %d clip %d chained gen produced %s (likely Veo safety filter); "
                            "retrying with hero-ref anchor",
                            seg.index, extra_idx, extra_result.output_type.value,
                        )
                        extra_result = await self._backend.generate(
                            segment=seg,
                            output_path=extra_output,
                            width=vid_cfg.width,
                            height=vid_cfg.height,
                            config=config,  # no chain_from → falls back to reference_image
                        )
                        if extra_result.output_type != VisualOutputType.VIDEO:
                            logger.warning(
                                "Segment %d clip %d hero-ref retry also failed; "
                                "stopping multi-clip gen with %d video clip(s). "
                                "Final muxed clip will be %.1fs short of audio.",
                                seg.index, extra_idx, len(clip_paths),
                                target_seconds - len(clip_paths) * CLIP_TARGET_SECONDS,
                            )
                            break
                    clip_paths.append(str(extra_result.path))

            if len(clip_paths) > 1:
                segment_clips[seg.index] = clip_paths
                logger.info(
                    "Segment %d: %d clips generated", seg.index, len(clip_paths),
                )

            logger.info(
                "Visual saved: %s (%s)",
                first_result.path.name, first_result.output_type.value,
            )

        # Tell assembly whether it's dealing with stills or pre-animated clips
        ctx.artifacts["visual_output_type"] = (
            VisualOutputType.VIDEO if has_video_clips else VisualOutputType.IMAGE
        )
        ctx.video.status = VideoStatus.VISUALS_DONE

        total_clips = sum(
            len(segment_clips.get(s.index, [s.image_path]))
            for s in ctx.script.segments
        )
        logger.info(
            "Visual generation complete: %d clips across %d segments via %s",
            total_clips, len(ctx.script.segments), self._backend.name,
        )
        return ctx


def _extract_last_frame(video_path: Path, output_path: Path) -> None:
    """Extract the final frame of a video as a PNG for Veo chain anchoring.

    Uses -sseof to seek a tiny bit before EOF, then writes one frame.
    -update 1 + -frames:v 1 ensures a single-image output. -q:v 1 keeps
    quality high since this PNG becomes the starting frame for the next
    Veo clip and we want it to look exactly like the moment we left off.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-sseof", "-0.1",
        "-i", str(video_path),
        "-update", "1",
        "-frames:v", "1",
        "-q:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Last-frame extract failed for {video_path.name}: {result.stderr}"
        )
