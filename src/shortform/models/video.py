"""Video model — the final assembled output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class VideoStatus(StrEnum):
    PENDING = "pending"
    SCRIPTED = "scripted"
    TTS_DONE = "tts_done"
    VISUALS_DONE = "visuals_done"
    ASSEMBLED = "assembled"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class Video:
    """A complete video and its production metadata."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    script_id: str = ""
    strategy_name: str = ""
    topic: str = ""
    title: str = ""
    status: VideoStatus = VideoStatus.PENDING
    output_path: str = ""
    duration: float = 0.0
    file_size_bytes: int = 0
    width: int = 1080
    height: int = 1920
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
