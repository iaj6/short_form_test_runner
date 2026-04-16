"""Tests for database operations."""

from pathlib import Path

import pytest

from shortform.models.script import Script, Segment
from shortform.models.video import Video, VideoStatus
from shortform.store.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close()


def test_save_and_get_video(db: Database):
    video = Video(strategy_name="test", title="Test Video", topic="testing")
    db.save_video(video)

    loaded = db.get_video(video.id)
    assert loaded is not None
    assert loaded.title == "Test Video"
    assert loaded.status == VideoStatus.PENDING


def test_list_videos(db: Database):
    for i in range(5):
        db.save_video(Video(strategy_name="test", title=f"Video {i}"))

    videos = db.list_videos()
    assert len(videos) == 5

    videos = db.list_videos(limit=3)
    assert len(videos) == 3


def test_save_and_get_script(db: Database):
    script = Script(
        strategy_name="test",
        topic="testing",
        title="Test Script",
        segments=[
            Segment(index=0, narration="Hello", visual_prompt="bg", text_overlay="hi"),
            Segment(index=1, narration="World", visual_prompt="bg2", text_overlay="earth"),
        ],
    )
    db.save_script(script, video_id="vid123")

    loaded = db.get_script(script.id)
    assert loaded is not None
    assert loaded.title == "Test Script"
    assert len(loaded.segments) == 2
    assert loaded.segments[0].narration == "Hello"


def test_checkpoints(db: Database):
    db.save_checkpoint("vid1", "script_gen", "completed")
    db.save_checkpoint("vid1", "tts", "in_progress")

    cp = db.get_checkpoint("vid1", "script_gen")
    assert cp is not None
    assert cp["status"] == "completed"

    last = db.get_last_completed_stage("vid1")
    assert last == "script_gen"
