"""Publish result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class Platform(StrEnum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"


@dataclass
class PublishResult:
    """Result of publishing a video to a platform."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    video_id: str = ""
    platform: Platform = Platform.YOUTUBE
    platform_id: str = ""  # platform-specific video/media ID
    url: str = ""
    status: str = "pending"
    error_message: str = ""
    published_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
