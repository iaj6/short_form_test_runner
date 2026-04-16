"""Pipeline framework — stage protocol, context, and runner."""

from shortform.pipeline.context import PipelineContext
from shortform.pipeline.runner import PipelineRunner
from shortform.pipeline.stage import PipelineStage

__all__ = ["PipelineStage", "PipelineContext", "PipelineRunner"]
