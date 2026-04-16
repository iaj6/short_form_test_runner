"""Local file asset management."""

from __future__ import annotations

import shutil
from pathlib import Path

from shortform.config import PROJECT_ROOT


class FileStore:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT / "data"
        self.videos_dir = self.base_dir / "videos"
        self.assets_dir = self.base_dir / "assets"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def video_dir(self, video_id: str) -> Path:
        """Get/create a directory for a specific video's working files."""
        d = self.assets_dir / video_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def segment_audio_path(self, video_id: str, segment_index: int) -> Path:
        return self.video_dir(video_id) / f"segment_{segment_index:02d}.mp3"

    def segment_image_path(self, video_id: str, segment_index: int) -> Path:
        return self.video_dir(video_id) / f"segment_{segment_index:02d}.png"

    def final_video_path(self, video_id: str, title: str = "") -> Path:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50].strip()
        filename = f"{video_id}_{safe_title}.mp4" if safe_title else f"{video_id}.mp4"
        return self.videos_dir / filename

    def cleanup_video_assets(self, video_id: str) -> None:
        """Remove working files for a video after successful assembly."""
        d = self.assets_dir / video_id
        if d.exists():
            shutil.rmtree(d)

    def get_video_size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0
