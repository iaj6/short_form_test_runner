"""Script and segment models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# Rough words-per-second for estimating a segment's spoken length before TTS
# measures the real thing. Deliberately backend-agnostic — Edge TTS lands near
# this, F5-TTS is much slower (~1.8 w/s, i.e. ~0.55s/word). Anything that
# budgets actual runtime (episode chunking for adapted scripts, deciding how
# many Veo clips a segment needs) should use the real per-backend rate instead
# of trusting this.
DEFAULT_WORDS_PER_SECOND = 2.5

# Silence between dialogue turns, for estimation only. The authoritative value
# at synthesis time is strategy.tts.turn_gap; this just has to be close enough
# that a twelve-exchange segment isn't wildly under-estimated.
DEFAULT_TURN_GAP = 0.28


@dataclass
class WordTiming:
    """Timing info for a single word from TTS."""

    word: str
    start: float  # seconds into the segment audio
    duration: float  # seconds


@dataclass
class TurnTiming:
    """Where one dialogue turn sits within its segment's assembled audio.

    Runtime-only (measured during TTS, never persisted). Exists so the visual
    stage can tell the video model WHO is speaking WHEN — without it, an
    image-to-video model has no idea anyone is talking and animates every
    puppet's mouth at once.
    """

    speaker: str
    start: float  # seconds into the segment audio
    duration: float


@dataclass
class Turn:
    """One speaker's line within a multi-speaker segment.

    Dialogue segments carry a list of these instead of relying on the segment's
    flat `narration`: TTSStage synthesizes each turn in that speaker's voice and
    concatenates the results into the segment's single audio file.
    """

    speaker: str
    line: str
    # Non-spoken performance note ("aside", "brandishing the phynance hook").
    # Never synthesized — it earns a longer pause before the line it annotates.
    stage_direction: str = ""
    # Named cue from data/sfx/library.yaml, played in the pause AFTER this line.
    # Referenced explicitly rather than inferred from the stage direction: most
    # directions ("aside", "aloud") describe delivery and make no sound at all.
    sfx: str = ""


@dataclass
class Segment:
    """A single segment of a video script."""

    index: int
    narration: str = ""
    visual_prompt: str = ""
    text_overlay: str = ""
    estimated_duration: float = 0.0  # seconds, estimated from text length
    actual_duration: float = 0.0  # seconds, measured after TTS

    # Paths populated during pipeline execution
    audio_path: str = ""
    image_path: str = ""

    # Hero variant key (set by VariantSelectionStage; consumed by VisualGenStage
    # to pick the per-segment reference image). Empty = use strategy default.
    hero_variant: str = ""

    # Dialogue turns for multi-speaker segments. Empty for single-narrator
    # strategies, which keep using `narration` directly. When present,
    # `narration` is DERIVED from the turns (see __post_init__) so every
    # existing consumer — caption alignment, variant selection, Veo prompt
    # building, the CLI script preview — keeps working without knowing that
    # speakers exist.
    turns: list[Turn] = field(default_factory=list)

    # Word-level timings from TTS (for animated subtitles)
    word_timings: list[WordTiming] = field(default_factory=list)

    # Per-turn timings measured during TTS, consumed by VisualGenStage to build
    # the per-clip speech schedule. Runtime-only, like word_timings.
    turn_timings: list[TurnTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Derive the flat narration from turns so `narration` is never empty on
        # a dialogue segment. An explicitly-supplied narration wins, which lets
        # an adaptation hand-write a cleaner flat text than the naive join.
        if self.turns and not self.narration:
            self.narration = self.flatten_turns()

    def flatten_turns(self) -> str:
        """The turns' spoken text as one string. Stage directions are excluded
        — they're performance notes, not narration."""
        return " ".join(t.line.strip() for t in self.turns if t.line.strip())

    @property
    def is_dialogue(self) -> bool:
        return bool(self.turns)

    @property
    def speakers(self) -> list[str]:
        """Distinct speakers, in order of first appearance.

        Lets visual selection compute who's on stage instead of inferring it
        from prose — the required hero variant for an adapted play is a
        function of (location × speakers present).
        """
        seen: list[str] = []
        for t in self.turns:
            if t.speaker not in seen:
                seen.append(t.speaker)
        return seen

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        """Rehydrate from a plain dict (script JSON, or the DB's segments_json).

        Exists because a bare `Segment(**data)` leaves nested turns as raw
        dicts, which then blow up somewhere much later than here.
        """
        known = {f for f in cls.__dataclass_fields__ if f not in {"turns", "word_timings"}}
        kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        kwargs["turns"] = [
            Turn(
                speaker=t.get("speaker", ""),
                line=t.get("line", ""),
                stage_direction=t.get("stage_direction", ""),
                sfx=t.get("sfx", ""),
            )
            for t in data.get("turns") or []
        ]
        return cls(**kwargs)

    def to_dict(self, include_runtime: bool = False) -> dict[str, Any]:
        """Plain-dict form for JSON/DB persistence.

        `include_runtime` adds the fields populated during a run (measured
        duration, asset paths) — the DB wants those, the editorial-gate script
        JSON does not.
        """
        data: dict[str, Any] = {
            "index": self.index,
            "narration": self.narration,
            "visual_prompt": self.visual_prompt,
            "text_overlay": self.text_overlay,
            "estimated_duration": self.estimated_duration,
            "hero_variant": self.hero_variant,
        }
        if self.turns:
            data["turns"] = [
                {"speaker": t.speaker, "line": t.line}
                | ({"stage_direction": t.stage_direction} if t.stage_direction else {})
                | ({"sfx": t.sfx} if t.sfx else {})
                for t in self.turns
            ]
        if include_runtime:
            data["actual_duration"] = self.actual_duration
            data["audio_path"] = self.audio_path
            data["image_path"] = self.image_path
        return data


def estimate_segment_duration(
    narration: str,
    n_turns: int = 0,
    words_per_second: float = DEFAULT_WORDS_PER_SECOND,
    turn_gap: float = DEFAULT_TURN_GAP,
) -> float:
    """Rough spoken duration for a segment, before TTS measures the real thing.

    Dialogue segments also pay the inter-turn silence, which stops being a
    rounding error once a segment holds a dozen short exchanges — a rapid-fire
    page of stichomythia is mostly pauses.
    """
    words = len(narration.split())
    speech = words / words_per_second if words_per_second > 0 else 0.0
    return speech + max(0, n_turns - 1) * turn_gap


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
    # Per-episode music mood, overriding the strategy's default. A serial wants
    # one recognisable score selected *per scene* — a strategy-level mood alone
    # would pin every episode to the same cue and leave the rest of the library
    # unused.
    music_mood: list[str] = field(default_factory=list)
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
            "music_mood": self.music_mood,
            "created_at": self.created_at.isoformat(),
            "segments": [s.to_dict() for s in self.segments],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_json(cls, path: Path) -> Script:
        """Load a Script from a JSON file. Recomputes estimated_duration from
        word count so edited narration gets up-to-date estimates.

        `narration` may be omitted on dialogue segments — it's derived from the
        turns. A segment with neither is a hard error rather than a silently
        silent segment.
        """
        data = json.loads(path.read_text())
        segments: list[Segment] = []
        for raw in data["segments"]:
            seg = Segment.from_dict(raw)
            if not seg.narration.strip():
                raise ValueError(
                    f"{path.name}: segment {raw.get('index', '?')} has neither "
                    "narration nor non-empty turns"
                )
            seg.estimated_duration = estimate_segment_duration(
                seg.narration, n_turns=len(seg.turns)
            )
            segments.append(seg)
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
            music_mood=[str(m) for m in data.get("music_mood") or []],
            created_at=created_at,
        )
