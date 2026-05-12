"""TTS backend protocol — the contract for text-to-speech backends."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from shortform.models.script import Segment, WordTiming


@dataclass
class TTSOutput:
    """Result of synthesizing audio for one segment."""

    audio_path: Path
    duration: float
    word_timings: list[WordTiming] = field(default_factory=list)


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol for TTS backends."""

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'edge', 'f5_tts')."""
        ...

    async def synthesize(
        self,
        segment: Segment,
        output_path: Path,
        config: dict[str, Any],
    ) -> TTSOutput:
        """Synthesize audio for a single segment.

        Args:
            segment: Script segment with narration text.
            output_path: Target file path. Backends write to this exact path
                         (convert codec if needed) so downstream stages see
                         a stable extension.
            config: Backend-specific config (merge of settings.tts + strategy.tts).

        Returns:
            TTSOutput with audio_path, duration, and (possibly empty) word_timings.
        """
        ...


def get_audio_duration(path: Path) -> float:
    """Probe audio duration via ffprobe (shared utility for backends)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        size = path.stat().st_size
        return size / 16000 if size > 0 else 0.0
