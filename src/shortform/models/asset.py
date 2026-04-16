"""Asset model for generated files (images, audio, etc.)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import uuid4


class AssetType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FONT = "font"
    MUSIC = "music"


@dataclass
class Asset:
    """A generated or sourced file used in video production."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    asset_type: AssetType = AssetType.IMAGE
    path: str = ""
    mime_type: str = ""
    size_bytes: int = 0
    duration: float = 0.0  # for audio/video assets
    width: int = 0
    height: int = 0
    video_id: str = ""
    segment_index: int = -1
    created_at: datetime = field(default_factory=datetime.now)
