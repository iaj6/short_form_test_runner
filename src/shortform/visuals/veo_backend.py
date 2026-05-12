"""Veo backend — Google's image-to-video generation via the Gemini API.

Workflow:
    1. Determine the base frame:
       - If strategy provides a `reference_image` path that exists, use it (anchors
         visual identity across every segment in a video — same character, same
         setting, same lighting). This is the path used by the Bartholomew /
         gothic_vignette strategy for character consistency.
       - Otherwise, generate a still base frame via PillowBackend (gradient).
    2. Animate it with Veo's image-to-video API, optionally seeded for cross-video
       consistency and optionally with a negative prompt to suppress unwanted
       elements.
    3. Return the video clip for assembly.

Requires a Google Gemini API key with Veo access.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from shortform.models.script import Segment
from shortform.visuals.backend import VisualOutput, VisualOutputType
from shortform.visuals.pillow_backend import PillowBackend

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 10


class VeoBackend:
    """Google Veo image-to-video backend."""

    def __init__(self, api_key: str = "", model: str = "veo-3.0-generate-001") -> None:
        self.api_key = api_key
        self.model = model

    @property
    def name(self) -> str:
        return "veo"

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
                "Veo requires a Google Gemini API key. "
                "Set GOOGLE_GEMINI_API_KEY in your .env file."
            )

        # Step 1: Resolve the base frame
        base_frame_path = await _resolve_base_frame(
            segment=segment,
            output_path=output_path,
            width=width,
            height=height,
            config=config,
        )

        # Step 2: Build animation prompt
        animation_prompt = _build_animation_prompt(
            segment.visual_prompt,
            config.get("animation_style", "cinematic slow push-in"),
        )

        # Step 3: Call Veo image-to-video
        video_path = output_path.with_suffix(".mp4")

        logger.info(
            "Animating segment %d with Veo: %s",
            segment.index,
            animation_prompt[:80],
        )

        try:
            await _generate_video(
                api_key=self.api_key,
                model=self.model,
                image_path=base_frame_path,
                prompt=animation_prompt,
                output_path=video_path,
                seed=config.get("veo_seed"),
                negative_prompt=config.get("veo_negative_prompt"),
            )
        except RuntimeError as e:
            # Safety filter or other Veo rejection — fall back to Pillow still image
            logger.warning(
                "Veo failed for segment %d, falling back to Pillow: %s",
                segment.index,
                e,
            )
            return await PillowBackend().generate(segment, output_path, width, height, config)

        return VisualOutput(
            path=video_path,
            output_type=VisualOutputType.VIDEO,
            width=width,
            height=height,
        )


async def _resolve_base_frame(
    segment: Segment,
    output_path: Path,
    width: int,
    height: int,
    config: dict[str, Any],
) -> Path:
    """Pick the still frame Veo will animate.

    Strategy can supply a `reference_image` path for visual identity consistency
    (same character/setting across every segment). When present and the file
    exists, that image is used as-is. Otherwise we generate a text-free Pillow
    gradient — text gets burned on in assembly to avoid Veo garbling rendered
    text during animation.
    """
    reference_image = config.get("reference_image")
    if reference_image:
        ref_path = Path(reference_image)
        if ref_path.exists():
            logger.info(
                "Segment %d using reference image: %s",
                segment.index,
                ref_path.name,
            )
            return ref_path
        logger.warning(
            "Reference image %s not found — falling back to Pillow gradient",
            ref_path,
        )

    text_free_segment = Segment(
        index=segment.index,
        narration=segment.narration,
        visual_prompt=segment.visual_prompt,
        text_overlay="",  # Veo would mangle rendered text
        estimated_duration=segment.estimated_duration,
        actual_duration=segment.actual_duration,
    )
    base = await PillowBackend().generate(
        text_free_segment, output_path, width, height, config
    )
    return base.path


def _build_animation_prompt(visual_prompt: str, animation_style: str) -> str:
    """Convert a visual description into cinematography language for Veo."""
    parts = []
    if animation_style:
        parts.append(animation_style)
    if visual_prompt:
        parts.append(visual_prompt)
    parts.append("smooth motion, high quality")
    return ", ".join(parts)


async def _generate_video(
    api_key: str,
    model: str,
    image_path: Path,
    prompt: str,
    output_path: Path,
    seed: int | None = None,
    negative_prompt: str | None = None,
) -> None:
    """Call the Veo image-to-video API and save the result."""
    client = genai.Client(api_key=api_key)
    image = types.Image.from_file(location=str(image_path))

    # Build config kwargs — only include optional fields when set so the SDK
    # doesn't see None for fields that don't accept it.
    #
    # Note: the public Gemini API does NOT support the `seed` parameter for
    # video generation (only the Vertex AI Veo endpoint does). The SDK
    # raises ValueError if we pass it, so we silently ignore `seed` from
    # strategy config until/unless we add a Vertex AI path. Visual identity
    # consistency relies on the reference_image anchor — per CLAUDE.md, the
    # reference image is the primary signal, "stronger than Veo seeds alone."
    config_kwargs: dict[str, Any] = {
        "aspect_ratio": "9:16",
        "number_of_videos": 1,
        "person_generation": "allow_adult",
        "resolution": "1080p",
    }
    if seed is not None:
        logger.debug(
            "veo_seed=%s present in config but ignored — Gemini API does not "
            "support seed for video generation",
            seed,
        )
    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image,
        config=types.GenerateVideosConfig(**config_kwargs),
    )

    # Poll until done
    while not operation.done:
        logger.debug("Waiting for Veo generation to complete...")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        operation = client.operations.get(operation)

    if not operation.response or not operation.response.generated_videos:
        # Log the full response for debugging — usually a safety filter rejection
        logger.error("Veo operation completed but returned no video. Response: %s", operation)
        raise RuntimeError(
            "Veo returned no video — likely blocked by safety filters. "
            f"Prompt was: {prompt[:120]}"
        )

    # Download and save the first result
    video = operation.response.generated_videos[0]
    client.files.download(file=video.video)
    video.video.save(str(output_path))

    logger.info("Veo video saved: %s", output_path.name)
