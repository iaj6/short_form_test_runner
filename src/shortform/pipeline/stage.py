"""Pipeline stage protocol — the contract every stage must implement."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from shortform.pipeline.context import PipelineContext


@runtime_checkable
class PipelineStage(Protocol):
    """Protocol for pipeline stages."""

    @property
    def name(self) -> str:
        """Unique stage identifier."""
        ...

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Run this stage, returning updated context."""
        ...

    def validate(self, ctx: PipelineContext) -> list[str]:
        """Check preconditions. Return list of error messages (empty = OK)."""
        ...
