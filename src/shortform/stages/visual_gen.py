"""Visual generation stage — delegates to pluggable backends."""

from __future__ import annotations

import logging

from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.visuals.backend import VisualBackend, VisualOutputType
from shortform.visuals.pillow_backend import PillowBackend

logger = logging.getLogger(__name__)


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

        for seg in ctx.script.segments:
            output_base = file_store.video_dir(ctx.video.id) / f"segment_{seg.index:02d}"
            logger.info(
                "Generating visual for segment %d [%s]", seg.index, self._backend.name
            )

            result = await self._backend.generate(
                segment=seg,
                output_path=output_base,
                width=vid_cfg.width,
                height=vid_cfg.height,
                config=config,
            )

            seg.image_path = str(result.path)

            # Track output type per segment for mixed Veo/Pillow fallback scenarios
            segment_types = ctx.artifacts.setdefault("segment_visual_types", {})
            segment_types[seg.index] = result.output_type

            if result.output_type == VisualOutputType.VIDEO:
                has_video_clips = True

            logger.info("Visual saved: %s (%s)", result.path.name, result.output_type.value)

        # Tell assembly whether it's dealing with stills or pre-animated clips
        ctx.artifacts["visual_output_type"] = (
            VisualOutputType.VIDEO if has_video_clips else VisualOutputType.IMAGE
        )
        ctx.video.status = VideoStatus.VISUALS_DONE

        logger.info(
            "Visual generation complete: %d assets via %s",
            len(ctx.script.segments),
            self._backend.name,
        )
        return ctx
