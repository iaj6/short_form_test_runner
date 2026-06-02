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

        Guards against two silent no-op modes that otherwise report success
        with no output: a resume_from that matches no stage (every stage gets
        skipped), and a resume_from naming the final stage (nothing runs).
        Both are turned into a clean FAILED result instead.
        """
        if resume_from is not None and resume_from not in {s.name for s in self.stages}:
            return self._fail(
                ctx,
                f"resume_from={resume_from!r} matches no stage "
                f"(stages: {', '.join(s.name for s in self.stages)})",
            )

        skipping = resume_from is not None
        ran_any = False

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
                ran_any = True
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

        if not ran_any:
            return self._fail(
                ctx,
                f"Pipeline ran zero stages (resume_from={resume_from!r} skipped "
                "every stage) — nothing to do",
            )

        ctx.video.completed_at = datetime.now()
        self.db.save_video(ctx.video)
        return ctx

    def _fail(self, ctx: PipelineContext, msg: str) -> PipelineContext:
        """Record a fatal pre/post-loop error and return a FAILED context."""
        logger.error(msg)
        ctx.errors.append(msg)
        ctx.video.status = VideoStatus.FAILED
        ctx.video.error_message = msg
        self.db.save_video(ctx.video)
        return ctx
