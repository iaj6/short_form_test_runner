"""Tests for script generation with mocked Claude responses."""

from unittest.mock import MagicMock, patch

import pytest

from shortform.config import AppSettings, StrategyConfig
from shortform.models.video import Video, VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.stages.script_gen import ScriptGenStage, _parse_script_response

MOCK_RESPONSE = """{
  "title": "The Power of Small Steps",
  "segments": [
    {
      "narration": "What if I told you that five minutes a day could change your entire life?",
      "visual_prompt": "Dark atmospheric background with glowing particles",
      "text_overlay": "5 minutes a day"
    },
    {
      "narration": "Tiny consistent habits compound over time. One push-up becomes fifty.",
      "visual_prompt": "Time-lapse of a seed growing into a tree",
      "text_overlay": "Tiny habits, massive results"
    },
    {
      "narration": "Start today. Start small. But start.",
      "visual_prompt": "Sunrise over mountains with golden light",
      "text_overlay": "Start today"
    }
  ]
}"""


@pytest.fixture
def strategy() -> StrategyConfig:
    return StrategyConfig(
        name="test_strategy",
        content={"segments": 3, "target_duration": 30},
        prompts={
            "system": "You are a motivational content creator.",
            "template": "Create a script about: {topic}\nSegments: {segments}",
        },
        topics=["small habits"],
    )


@pytest.fixture
def settings() -> AppSettings:
    s = AppSettings()
    s.anthropic_api_key = "sk-test-key"
    return s


@pytest.mark.asyncio
async def test_script_gen_parses_response(strategy: StrategyConfig, settings: AppSettings):
    stage = ScriptGenStage()
    video = Video()
    ctx = PipelineContext(settings=settings, strategy=strategy, video=video)

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=MOCK_RESPONSE)]

    with patch("shortform.stages.script_gen.anthropic.Anthropic") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        mock_client_cls.return_value = mock_client

        result = await stage.execute(ctx)

    assert result.script.title == "The Power of Small Steps"
    assert len(result.script.segments) == 3
    assert result.script.segments[0].text_overlay == "5 minutes a day"
    assert result.video.status == VideoStatus.SCRIPTED
    assert result.script.total_duration > 0


def test_script_gen_validates_api_key(strategy: StrategyConfig):
    stage = ScriptGenStage()
    settings = AppSettings(anthropic_api_key="")  # no API key
    ctx = PipelineContext(settings=settings, strategy=strategy)
    errors = stage.validate(ctx)
    assert any("ANTHROPIC_API_KEY" in e for e in errors)


# --- _parse_script_response hardening (#6) ------------------------------------


def test_parse_strips_markdown_fences():
    fenced = "```json\n" + MOCK_RESPONSE + "\n```"
    script = _parse_script_response(fenced, "test_strategy", "topic")
    assert script.title == "The Power of Small Steps"
    assert len(script.segments) == 3


def test_parse_invalid_json_raises_descriptive_error():
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _parse_script_response("Sure! Here is your script: (no json)", "s", "t")


def test_parse_truncated_json_raises():
    # max_tokens cutoff mid-object — json.loads would raise a bare decode error.
    truncated = '{"title": "X", "segments": [{"narration": "half a sen'
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _parse_script_response(truncated, "s", "t")


def test_parse_missing_segments_raises():
    with pytest.raises(RuntimeError, match="segments"):
        _parse_script_response('{"title": "X"}', "s", "t")


def test_parse_empty_segments_raises():
    with pytest.raises(RuntimeError, match="segments"):
        _parse_script_response('{"title": "X", "segments": []}', "s", "t")


def test_parse_segment_missing_narration_raises():
    bad = '{"title": "X", "segments": [{"visual_prompt": "bg"}]}'
    with pytest.raises(RuntimeError, match="narration"):
        _parse_script_response(bad, "s", "t")


def test_parse_non_object_json_raises():
    with pytest.raises(RuntimeError, match="not an object"):
        _parse_script_response('["a", "b"]', "s", "t")
