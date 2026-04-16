"""Grok Imagine backend — xAI's image-to-video generation.

Uses the image-first workflow recommended for best results:
1. Generate a still frame (via Pillow or other image source)
2. Animate it with Grok Imagine's image-to-video API

Requires an xAI API key. See: https://docs.x.ai/docs/guides/image-and-video-generation

Pricing: ~$4.20/min of generated video (as of March 2026).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from shortform.models.script import Segment
from shortform.visuals.backend import VisualOutput, VisualOutputType

logger = logging.getLogger(__name__)


class GrokImagineBackend:
    """Grok Imagine image-to-video backend.

    Workflow:
        1. Generate a base frame using PillowBackend (composition/text)
        2. Send to Grok Imagine with a cinematography-style prompt
        3. Receive animated video clip
    """

    def __init__(self, api_key: str = "", base_url: str = "https://api.x.ai/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url

    @property
    def name(self) -> str:
        return "grok_imagine"

    async def generate(
        self,
        segment: Segment,
        output_path: Path,
        width: int,
        height: int,
        config: dict[str, Any],
    ) -> VisualOutput:
        if not self.api_key:
            raise RuntimeError(
                "Grok Imagine requires an xAI API key. "
                "Set XAI_API_KEY in your .env file. "
                "Subscribe at https://x.ai to get API access."
            )

        # Step 1: Generate base frame with Pillow
        from shortform.visuals.pillow_backend import PillowBackend

        pillow = PillowBackend()
        base_frame = await pillow.generate(segment, output_path, width, height, config)

        # Step 2: Build cinematography-style prompt from visual_prompt
        animation_prompt = _build_animation_prompt(
            segment.visual_prompt,
            config.get("animation_style", "cinematic slow push-in"),
        )

        # Step 3: Send to Grok Imagine image-to-video API
        video_path = output_path.with_suffix(".mp4")
        duration = segment.actual_duration or segment.estimated_duration or 5.0

        logger.info(
            "Animating segment %d with Grok Imagine (%.1fs): %s",
            segment.index,
            duration,
            animation_prompt[:80],
        )

        await _call_grok_image_to_video(
            api_key=self.api_key,
            base_url=self.base_url,
            image_path=base_frame.path,
            prompt=animation_prompt,
            output_path=video_path,
            duration=duration,
            width=width,
            height=height,
        )

        return VisualOutput(
            path=video_path,
            output_type=VisualOutputType.VIDEO,
            duration=duration,
            width=width,
            height=height,
        )


def _build_animation_prompt(visual_prompt: str, animation_style: str) -> str:
    """Convert a visual description into cinematography language.

    Grok Imagine responds best to filmmaking terminology:
    camera movements, lighting descriptors, lens choices.
    """
    parts = []
    if animation_style:
        parts.append(animation_style)
    if visual_prompt:
        parts.append(visual_prompt)
    parts.append("smooth motion, high quality, 4K")
    return ", ".join(parts)


async def _call_grok_image_to_video(
    api_key: str,
    base_url: str,
    image_path: Path,
    prompt: str,
    output_path: Path,
    duration: float,
    width: int,
    height: int,
) -> None:
    """Call the xAI Grok Imagine image-to-video API.

    This is a stub — the actual API integration will be implemented
    once xAI API access is set up. The API is expected to follow
    the pattern documented at https://docs.x.ai/docs/guides/image-and-video-generation

    Expected flow:
        1. POST image + prompt to create a generation task
        2. Poll for completion
        3. Download the resulting video
    """
    # TODO: Implement when xAI API access is available
    #
    # Expected API shape (based on xAI docs):
    #
    # import httpx
    # import base64
    #
    # image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    #
    # async with httpx.AsyncClient() as client:
    #     # Create generation
    #     resp = await client.post(
    #         f"{base_url}/images/generations",
    #         headers={"Authorization": f"Bearer {api_key}"},
    #         json={
    #             "model": "grok-2-image",
    #             "prompt": prompt,
    #             "image": image_b64,
    #             "response_format": "url",
    #             "n": 1,
    #         },
    #     )
    #     resp.raise_for_status()
    #     video_url = resp.json()["data"][0]["url"]
    #
    #     # Download video
    #     video_resp = await client.get(video_url)
    #     output_path.write_bytes(video_resp.content)

    raise NotImplementedError(
        "Grok Imagine API integration not yet implemented. "
        "Waiting for xAI API access. Use --visual-backend pillow for now."
    )
