"""Analytics stage — placeholder for Phase 3."""

from __future__ import annotations

import logging

from shortform.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


class AnalyticsStage:
    @property
    def name(self) -> str:
        return "analytics"

    def validate(self, ctx: PipelineContext) -> list[str]:
        return []  # Phase 3

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("Analytics stage not yet implemented (Phase 3)")
        return ctx
