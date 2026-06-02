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


@pytest.mark.asyncio
async def test_veo_uses_reference_image_when_present(tmp_path: Path):
    """When config provides a reference_image that exists, Veo animates that
    image directly — no Pillow gradient is generated."""
    backend = VeoBackend(api_key="test-key")
    segment = Segment(
        index=0, narration="Test", visual_prompt="dim study", text_overlay=""
    )

    # Create a real reference image file on disk
    ref_image = tmp_path / "bartholomew_hero.png"
    # Use Pillow to generate a real PNG (Veo SDK loads it via types.Image.from_file)
    from PIL import Image
    Image.new("RGB", (1080, 1920), (10, 8, 8)).save(ref_image)

    captured: dict = {}

    async def fake_generate_video(
        api_key, model, image_path, prompt, output_path, seed=None, negative_prompt=None
    ):
        captured["image_path"] = image_path
        captured["seed"] = seed
        captured["negative_prompt"] = negative_prompt
        Path(output_path).write_bytes(b"fake-mp4")

    with patch("shortform.visuals.veo_backend._generate_video", side_effect=fake_generate_video):
        with patch("shortform.visuals.veo_backend.PillowBackend") as mock_pillow:
            await backend.generate(
                segment=segment,
                output_path=tmp_path / "seg",
                width=1080,
                height=1920,
                config={
                    "reference_image": str(ref_image),
                    "veo_seed": 314159,
                    "veo_negative_prompt": "people, faces, modern clothes",
                },
            )

            mock_pillow.assert_not_called()  # reference image used directly

    assert captured["image_path"] == ref_image
    assert captured["seed"] == 314159
    assert captured["negative_prompt"] == "people, faces, modern clothes"


@pytest.mark.asyncio
async def test_veo_falls_back_to_pillow_when_reference_image_missing(tmp_path: Path):
    """When reference_image path is set but the file doesn't exist, log a
    warning and fall back to Pillow gradient (don't crash the pipeline)."""
    backend = VeoBackend(api_key="test-key")
    segment = Segment(
        index=0, narration="Test", visual_prompt="bg", text_overlay=""
    )

    async def fake_generate_video(*args, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"fake-mp4")

    with patch("shortform.visuals.veo_backend._generate_video", side_effect=fake_generate_video):
        output = await backend.generate(
            segment=segment,
            output_path=tmp_path / "seg",
            width=1080,
            height=1920,
            config={"reference_image": str(tmp_path / "does_not_exist.png")},
        )

    # Pillow gradient base frame should have been written alongside the mp4
    assert (tmp_path / "seg.png").exists()
    assert output.output_type == VisualOutputType.VIDEO


@pytest.mark.asyncio
async def test_veo_poll_loop_times_out(tmp_path: Path):
    """A generation operation that never reports done must raise once the
    wall-clock deadline passes — not spin forever."""
    import itertools

    from PIL import Image as PILImage

    from shortform.visuals import veo_backend

    ref_image = tmp_path / "ref.png"
    PILImage.new("RGB", (64, 64), (0, 0, 0)).save(ref_image)

    stuck_op = MagicMock()
    stuck_op.done = False  # never completes
    mock_client = MagicMock()
    mock_client.models.generate_videos.return_value = stuck_op
    mock_client.operations.get.return_value = stuck_op

    # First monotonic() call sets the deadline; every subsequent call is far in
    # the future so the very first poll-loop check trips the deadline.
    clock = itertools.chain([0.0], itertools.repeat(1e9))

    with patch("shortform.visuals.veo_backend.genai.Client", return_value=mock_client), \
         patch("shortform.visuals.veo_backend.time.monotonic", side_effect=lambda: next(clock)), \
         patch("shortform.visuals.veo_backend.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(RuntimeError, match="deadline"):
            await veo_backend._generate_video(
                api_key="test-key",
                model="veo-3.1-generate-preview",
                image_path=ref_image,
                prompt="a cat",
                output_path=tmp_path / "out.mp4",
            )
