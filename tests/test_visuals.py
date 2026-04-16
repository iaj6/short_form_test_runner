"""Tests for the pluggable visual backend system."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shortform.models.script import Segment
from shortform.visuals import get_backend, list_backends
from shortform.visuals.backend import VisualBackend, VisualOutputType
from shortform.visuals.pillow_backend import PillowBackend
from shortform.visuals.veo_backend import VeoBackend


def test_list_backends():
    backends = list_backends()
    assert "pillow" in backends
    assert "veo" in backends


def test_get_backend_pillow():
    backend = get_backend("pillow")
    assert isinstance(backend, PillowBackend)
    assert backend.name == "pillow"


def test_get_backend_veo():
    backend = get_backend("veo")
    assert isinstance(backend, VeoBackend)
    assert backend.name == "veo"


def test_get_backend_unknown():
    with pytest.raises(ValueError, match="Unknown visual backend"):
        get_backend("nonexistent")


def test_pillow_implements_protocol():
    assert isinstance(PillowBackend(), VisualBackend)


def test_veo_implements_protocol():
    assert isinstance(VeoBackend(), VisualBackend)


@pytest.mark.asyncio
async def test_pillow_generates_image(tmp_path: Path):
    backend = PillowBackend()
    segment = Segment(
        index=0,
        narration="Test narration",
        visual_prompt="dark background",
        text_overlay="Test Overlay",
    )

    output = await backend.generate(
        segment=segment,
        output_path=tmp_path / "test_segment",
        width=540,
        height=960,
        config={"font_size": 32, "gradient_top": "#000000", "gradient_bottom": "#333333"},
    )

    assert output.path.exists()
    assert output.path.suffix == ".png"
    assert output.output_type == VisualOutputType.IMAGE
    assert output.width == 540
    assert output.height == 960


@pytest.mark.asyncio
async def test_veo_fails_without_api_key(tmp_path: Path):
    backend = VeoBackend()  # no API key
    segment = Segment(
        index=0, narration="Test", visual_prompt="bg", text_overlay="hi"
    )

    with pytest.raises(RuntimeError, match="Google Gemini API key"):
        await backend.generate(
            segment=segment,
            output_path=tmp_path / "test",
            width=1080,
            height=1920,
            config={},
        )


@pytest.mark.asyncio
async def test_veo_calls_api_with_correct_params(tmp_path: Path):
    """Test that VeoBackend correctly calls the Veo API (mocked)."""
    backend = VeoBackend(api_key="test-key")
    segment = Segment(
        index=0,
        narration="Test narration",
        visual_prompt="dramatic sky",
        text_overlay="Hello",
        estimated_duration=6.0,
    )

    # Mock the Veo API call, let Pillow run for real
    mock_video = MagicMock()
    mock_video.video = MagicMock()
    mock_operation = MagicMock()
    mock_operation.done = True
    mock_operation.response.generated_videos = [mock_video]

    mock_client = MagicMock()
    mock_client.models.generate_videos.return_value = mock_operation

    with patch("shortform.visuals.veo_backend.genai.Client", return_value=mock_client):
        # Write a dummy mp4 when save is called
        def fake_save(path: str) -> None:
            Path(path).write_bytes(b"fake-mp4-data")

        mock_video.video.save.side_effect = fake_save

        output = await backend.generate(
            segment=segment,
            output_path=tmp_path / "test_segment",
            width=1080,
            height=1920,
            config={},
        )

    assert output.output_type == VisualOutputType.VIDEO
    assert output.path.suffix == ".mp4"
    mock_client.models.generate_videos.assert_called_once()
