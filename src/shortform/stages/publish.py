"""Publish stage — placeholder for Phase 2."""

from __future__ import annotations

import logging

from shortform.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class PublishStage:
    @property
    def name(self) -> str:
        return "publish"

    def validate(self, ctx: PipelineContext) -> list[str]:
        return []  # Phase 2

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("Publish stage not yet implemented (Phase 2)")
        return ctx
