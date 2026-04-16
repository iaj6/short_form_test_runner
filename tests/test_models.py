"""Tests for data models."""

from shortform.models.script import Script, Segment
from shortform.models.video import Video, VideoStatus


def test_script_full_narration():
    script = Script(
        segments=[
            Segment(index=0, narration="Hello world", visual_prompt="", text_overlay=""),
            Segment(index=1, narration="Goodbye world", visual_prompt="", text_overlay=""),
        ]
    )
    assert script.full_narration == "Hello world Goodbye world"
    assert script.segment_count == 2


def test_video_defaults():
    video = Video()
    assert video.status == VideoStatus.PENDING
    assert video.width == 1080
    assert video.height == 1920
    assert len(video.id) == 12
