"""Script and segment models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class WordTiming:
    """Timing info for a single word from TTS."""

    word: str
    start: float  # seconds into the segment audio
    duration: float  # seconds


@dataclass
class Segment:
    """A single segment of a video script."""

    index: int
    narration: str
    visual_prompt: str
    text_overlay: str
    estimated_duration: float = 0.0  # seconds, estimated from text length
    actual_duration: float = 0.0  # seconds, measured after TTS

    # Paths populated during pipeline execution
    audio_path: str = ""
    image_path: str = ""

    # Word-level timings from TTS (for animated subtitles)
    word_timings: list[WordTiming] = field(default_factory=list)


@dataclass
class Script:
    """Complete video script with metadata."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    strategy_name: str = ""
    topic: str = ""
    title: str = ""
    segments: list[Segment] = field(default_factory=list)
    total_duration: float = 0.0
    raw_llm_response: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def full_narration(self) -> str:
        return " ".join(s.narration for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)
