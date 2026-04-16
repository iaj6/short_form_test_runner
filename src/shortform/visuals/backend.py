"""Visual backend protocol — the contract for visual generation backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from shortform.models.script import Segment


class VisualOutputType(StrEnum):
    IMAGE = "image"  # Still frame (needs Ken Burns in assembly)
    VIDEO = "video"  # Already animated clip (skip Ken Burns in assembly)


@dataclass
class VisualOutput:
    """Result of generating a visual for one segment."""

    path: Path
    output_type: VisualOutputType
    duration: float = 0.0  # Only relevant for video outputs
    width: int = 0
    height: int = 0


@runtime_checkable
class VisualBackend(Protocol):
    """Protocol for visual generation backends."""

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'pillow', 'veo')."""
        ...

    async def generate(
        self,
        segment: Segment,
        output_path: Path,
        width: int,
        height: int,
        config: dict[str, Any],
    ) -> VisualOutput:
        """Generate a visual for a single segment.

        Args:
            segment: The script segment to visualize.
            output_path: Where to write the output file (without extension —
                         backend appends .png or .mp4 as appropriate).
            width: Target width in pixels.
            height: Target height in pixels.
            config: Backend-specific config (from settings.visuals + strategy overrides).

        Returns:
            VisualOutput with path, type, and metadata.
        """
        ...
