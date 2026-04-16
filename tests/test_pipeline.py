"""Tests for the pipeline framework."""

from pathlib import Path

import pytest

from shortform.models.video import Video, VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.pipeline.runner import PipelineRunner
from shortform.store.db import Database


class PassStage:
    def __init__(self, stage_name: str = "pass"):
        self._name = stage_name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, ctx: PipelineContext) -> list[str]:
        return []

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        ctx.artifacts[self._name] = "done"
        return ctx


class FailStage:
    @property
    def name(self) -> str:
        return "fail"

    def validate(self, ctx: PipelineContext) -> list[str]:
        return []

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        raise RuntimeError("Stage failed intentionally")


class ValidationFailStage:
    @property
    def name(self) -> str:
        return "val_fail"

    def validate(self, ctx: PipelineContext) -> list[str]:
        return ["Missing required input"]

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        return ctx


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close()


@pytest.mark.asyncio
async def test_pipeline_runs_stages(db: Database):
    stages = [PassStage("a"), PassStage("b"), PassStage("c")]
    runner = PipelineRunner(stages=stages, db=db)

    video = Video()
    db.save_video(video)
    ctx = PipelineContext(video=video)

    result = await runner.run(ctx)
    assert result.completed_stages == ["a", "b", "c"]
    assert result.artifacts == {"a": "done", "b": "done", "c": "done"}
    assert not result.errors


@pytest.mark.asyncio
async def test_pipeline_handles_failure(db: Database):
    stages = [PassStage("a"), FailStage(), PassStage("c")]
    runner = PipelineRunner(stages=stages, db=db)

    video = Video()
    db.save_video(video)
    ctx = PipelineContext(video=video)

    result = await runner.run(ctx)
    assert result.completed_stages == ["a"]
    assert result.video.status == VideoStatus.FAILED
    assert len(result.errors) == 1


@pytest.mark.asyncio
async def test_pipeline_validation_failure(db: Database):
    stages = [PassStage("a"), ValidationFailStage()]
    runner = PipelineRunner(stages=stages, db=db)

    video = Video()
    db.save_video(video)
    ctx = PipelineContext(video=video)

    result = await runner.run(ctx)
    assert result.completed_stages == ["a"]
    assert result.video.status == VideoStatus.FAILED
