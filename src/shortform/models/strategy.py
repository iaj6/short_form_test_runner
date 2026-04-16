"""Strategy record for tracking strategy performance over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class StrategyRecord:
    """Tracks a strategy's usage and performance history."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    name: str = ""
    category: str = ""
    parent_id: str | None = None  # lineage tracking for mutations
    generation: int = 0
    total_videos: int = 0
    avg_score: float = 0.0
    best_score: float = 0.0
    last_used: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
