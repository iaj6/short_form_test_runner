"""Pipeline runner — orchestrates stages with checkpointing."""

from __future__ import annotations

import logging
from datetime import datetime

from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.pipeline.stage import PipelineStage
from shortform.store.db import Database

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs a sequence of pipeline stages with validation and checkpoints."""

    def __init__(self, stages: list[PipelineStage], db: Database) -> None:
        self.stages = stages
        self.db = db

    async def run(self, ctx: PipelineContext, resume_from: str | None = None) -> PipelineContext:
        """Execute all stages in sequence.

        If resume_from is provided, skip stages up to and including that stage.
        """
        skipping = resume_from is not None

        for stage in self.stages:
            if skipping:
                if stage.name == resume_from:
                    skipping = False
                logger.info("Skipping completed stage: %s", stage.name)
                continue

            # Validate preconditions
            errors = stage.validate(ctx)
            if errors:
                msg = f"Stage {stage.name} validation failed: {'; '.join(errors)}"
                logger.error(msg)
                ctx.errors.append(msg)
                ctx.video.status = VideoStatus.FAILED
                ctx.video.error_message = msg
                self.db.save_video(ctx.video)
                return ctx

            # Checkpoint: starting
            logger.info("Starting stage: %s", stage.name)
            self.db.save_checkpoint(ctx.video.id, stage.name, "in_progress")

            try:
                ctx = await stage.execute(ctx)
                ctx.mark_stage_complete(stage.name)
                self.db.save_checkpoint(ctx.video.id, stage.name, "completed")
                self.db.save_video(ctx.video)
                logger.info("Completed stage: %s", stage.name)
            except Exception as e:
                msg = f"Stage {stage.name} failed: {e}"
                logger.error(msg, exc_info=True)
                ctx.errors.append(msg)
                ctx.video.status = VideoStatus.FAILED
                ctx.video.error_message = msg
                self.db.save_checkpoint(ctx.video.id, stage.name, "failed", error_message=str(e))
                self.db.save_video(ctx.video)
                return ctx

        ctx.video.completed_at = datetime.now()
        self.db.save_video(ctx.video)
        return ctx
