"""SQLite database for videos, scripts, and pipeline state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from shortform.models.script import Script, Segment
from shortform.models.video import Video, VideoStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    script_id TEXT,
    strategy_name TEXT,
    topic TEXT,
    title TEXT,
    status TEXT DEFAULT 'pending',
    output_path TEXT,
    duration REAL DEFAULT 0,
    file_size_bytes INTEGER DEFAULT 0,
    width INTEGER DEFAULT 1080,
    height INTEGER DEFAULT 1920,
    error_message TEXT DEFAULT '',
    created_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS scripts (
    id TEXT PRIMARY KEY,
    video_id TEXT,
    strategy_name TEXT,
    topic TEXT,
    title TEXT,
    segments_json TEXT,
    total_duration REAL DEFAULT 0,
    raw_llm_response TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_checkpoints (
    video_id TEXT,
    stage_name TEXT,
    status TEXT DEFAULT 'pending',
    context_json TEXT,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    PRIMARY KEY (video_id, stage_name)
);

CREATE TABLE IF NOT EXISTS strategy_records (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE,
    category TEXT,
    parent_id TEXT,
    generation INTEGER DEFAULT 0,
    total_videos INTEGER DEFAULT 0,
    avg_score REAL DEFAULT 0,
    best_score REAL DEFAULT 0,
    last_used TEXT,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_strategy ON videos(strategy_name);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_scripts_video ON scripts(video_id);
"""


class Database:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def initialize(self) -> None:
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Videos ---

    def save_video(self, video: Video) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO videos
            (id, script_id, strategy_name, topic, title, status, output_path,
             duration, file_size_bytes, width, height, error_message,
             created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                video.id,
                video.script_id,
                video.strategy_name,
                video.topic,
                video.title,
                video.status.value,
                video.output_path,
                video.duration,
                video.file_size_bytes,
                video.width,
                video.height,
                video.error_message,
                video.created_at.isoformat(),
                video.completed_at.isoformat() if video.completed_at else None,
            ),
        )
        self.conn.commit()

    def get_video(self, video_id: str) -> Video | None:
        row = self.conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
        if not row:
            return None
        return Video(
            id=row["id"],
            script_id=row["script_id"] or "",
            strategy_name=row["strategy_name"] or "",
            topic=row["topic"] or "",
            title=row["title"] or "",
            status=VideoStatus(row["status"]),
            output_path=row["output_path"] or "",
            duration=row["duration"] or 0.0,
            file_size_bytes=row["file_size_bytes"] or 0,
            width=row["width"] or 1080,
            height=row["height"] or 1920,
            error_message=row["error_message"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
        )

    def list_videos(
        self, strategy: str | None = None, status: VideoStatus | None = None, limit: int = 50
    ) -> list[Video]:
        query = "SELECT * FROM videos WHERE 1=1"
        params: list[str] = []
        if strategy:
            query += " AND strategy_name = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(str(limit))
        rows = self.conn.execute(query, params).fetchall()
        return [self.get_video(row["id"]) for row in rows if self.get_video(row["id"])]  # type: ignore[misc]

    # --- Scripts ---

    def save_script(self, script: Script, video_id: str) -> None:
        segments_data = [
            {
                "index": s.index,
                "narration": s.narration,
                "visual_prompt": s.visual_prompt,
                "text_overlay": s.text_overlay,
                "estimated_duration": s.estimated_duration,
                "actual_duration": s.actual_duration,
                "audio_path": s.audio_path,
                "image_path": s.image_path,
            }
            for s in script.segments
        ]
        self.conn.execute(
            """INSERT OR REPLACE INTO scripts
            (id, video_id, strategy_name, topic, title, segments_json,
             total_duration, raw_llm_response, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                script.id,
                video_id,
                script.strategy_name,
                script.topic,
                script.title,
                json.dumps(segments_data),
                script.total_duration,
                script.raw_llm_response,
                script.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_script(self, script_id: str) -> Script | None:
        row = self.conn.execute("SELECT * FROM scripts WHERE id = ?", (script_id,)).fetchone()
        if not row:
            return None
        segments_data = json.loads(row["segments_json"] or "[]")
        segments = [Segment(**s) for s in segments_data]
        return Script(
            id=row["id"],
            strategy_name=row["strategy_name"] or "",
            topic=row["topic"] or "",
            title=row["title"] or "",
            segments=segments,
            total_duration=row["total_duration"] or 0.0,
            raw_llm_response=row["raw_llm_response"] or "",
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # --- Checkpoints ---

    def save_checkpoint(
        self,
        video_id: str,
        stage_name: str,
        status: str,
        context_json: str = "",
        error_message: str = "",
    ) -> None:
        now = datetime.now().isoformat()
        completed = now if status == "completed" else None
        self.conn.execute(
            """INSERT OR REPLACE INTO pipeline_checkpoints
            (video_id, stage_name, status, context_json, started_at, completed_at, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (video_id, stage_name, status, context_json, now, completed, error_message),
        )
        self.conn.commit()

    def get_checkpoint(self, video_id: str, stage_name: str) -> dict[str, str] | None:
        row = self.conn.execute(
            "SELECT * FROM pipeline_checkpoints WHERE video_id = ? AND stage_name = ?",
            (video_id, stage_name),
        ).fetchone()
        if not row:
            return None
        return dict(row)

    def get_last_completed_stage(self, video_id: str) -> str | None:
        row = self.conn.execute(
            """SELECT stage_name FROM pipeline_checkpoints
            WHERE video_id = ? AND status = 'completed'
            ORDER BY completed_at DESC LIMIT 1""",
            (video_id,),
        ).fetchone()
        return row["stage_name"] if row else None
