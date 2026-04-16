"""Pipeline context — accumulates artifacts as it flows through stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shortform.config import AppSettings, StrategyConfig
from shortform.models.script import Script
from shortform.models.video import Video


@dataclass
class PipelineContext:
    """Shared state passed through all pipeline stages."""

    # Configuration
    settings: AppSettings = field(default_factory=AppSettings)
    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(name="default"))

    # Core artifacts (populated by stages)
    video: Video = field(default_factory=Video)
    script: Script = field(default_factory=Script)

    # Selected topic for this run
    topic: str = ""

    # Stage-specific data (extensible)
    artifacts: dict[str, Any] = field(default_factory=dict)

    # Tracking
    completed_stages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def mark_stage_complete(self, stage_name: str) -> None:
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)

    def has_completed(self, stage_name: str) -> bool:
        return stage_name in self.completed_stages
