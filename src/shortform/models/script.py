"""Script and segment models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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

    # Hero variant key (set by VariantSelectionStage; consumed by VisualGenStage
    # to pick the per-segment reference image). Empty = use strategy default.
    hero_variant: str = ""

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

    def save_json(self, path: Path) -> None:
        """Serialize this Script to JSON at `path`.

        Skips raw_llm_response (large, not user-editable) and runtime-only
        fields (audio_path, image_path, actual_duration, word_timings) since
        those get populated by TTS/visual_gen stages.
        """
        data = {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "topic": self.topic,
            "title": self.title,
            "total_duration": self.total_duration,
            "created_at": self.created_at.isoformat(),
            "segments": [
                {
                    "index": s.index,
                    "narration": s.narration,
                    "visual_prompt": s.visual_prompt,
                    "text_overlay": s.text_overlay,
                    "estimated_duration": s.estimated_duration,
                    "hero_variant": s.hero_variant,
                }
                for s in self.segments
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_json(cls, path: Path) -> Script:
        """Load a Script from a JSON file. Recomputes estimated_duration from
        word count so edited narration gets up-to-date estimates."""
        data = json.loads(path.read_text())
        segments = [
            Segment(
                index=s["index"],
                narration=s["narration"],
                visual_prompt=s.get("visual_prompt", ""),
                text_overlay=s.get("text_overlay", ""),
                estimated_duration=len(s["narration"].split()) / 2.5,
                hero_variant=s.get("hero_variant", ""),
            )
            for s in data["segments"]
        ]
        created_raw = data.get("created_at")
        created_at = (
            datetime.fromisoformat(created_raw)
            if created_raw
            else datetime.now()
        )
        return cls(
            id=data.get("id", uuid4().hex[:12]),
            strategy_name=data.get("strategy_name", ""),
            topic=data.get("topic", ""),
            title=data.get("title", ""),
            segments=segments,
            total_duration=data.get("total_duration", 0.0),
            created_at=created_at,
        )
