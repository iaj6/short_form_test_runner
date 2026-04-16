"""Script generation stage — Claude API produces a narration script."""

from __future__ import annotations

import json
import logging
import random

import anthropic

from shortform.models.script import Script, Segment
from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)

RESPONSE_FORMAT_INSTRUCTION = """
Respond with valid JSON only. Use this exact structure:
{
  "title": "Short catchy title for the video",
  "segments": [
    {
      "narration": "The spoken text for this segment",
      "visual_prompt": "Description of the background visual",
      "text_overlay": "Key phrase shown on screen"
    }
  ]
}
"""


class ScriptGenStage:
    @property
    def name(self) -> str:
        return "script_gen"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.settings.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY not configured")
        if not ctx.strategy.prompts.get("system"):
            errors.append("Strategy missing 'system' prompt")
        if not ctx.strategy.prompts.get("template"):
            errors.append("Strategy missing 'template' prompt")
        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        # Pick topic
        topic = ctx.topic
        if not topic and ctx.strategy.topics:
            topic = random.choice(ctx.strategy.topics)
        ctx.topic = topic

        num_segments = ctx.strategy.content.get("segments", 3)
        user_prompt = ctx.strategy.prompts["template"].format(
            topic=topic,
            segments=num_segments,
        )
        user_prompt += "\n" + RESPONSE_FORMAT_INSTRUCTION

        logger.info("Generating script for topic: %s", topic)

        client = anthropic.Anthropic(api_key=ctx.settings.anthropic_api_key)
        message = client.messages.create(
            model=ctx.settings.llm.model,
            max_tokens=ctx.settings.llm.max_tokens,
            temperature=ctx.settings.llm.temperature,
            system=ctx.strategy.prompts["system"],
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw_text = message.content[0].text  # type: ignore[union-attr]
        logger.debug("Raw LLM response: %s", raw_text)

        # Parse response
        script = _parse_script_response(raw_text, ctx.strategy.name, topic)
        script.raw_llm_response = raw_text

        # Estimate durations from word count (~150 words/min = 2.5 words/sec)
        for seg in script.segments:
            word_count = len(seg.narration.split())
            seg.estimated_duration = word_count / 2.5

        script.total_duration = sum(s.estimated_duration for s in script.segments)

        ctx.script = script
        ctx.video.script_id = script.id
        ctx.video.strategy_name = ctx.strategy.name
        ctx.video.topic = topic
        ctx.video.title = script.title
        ctx.video.status = VideoStatus.SCRIPTED

        logger.info(
            "Script generated: %d segments, ~%.1fs estimated",
            script.segment_count,
            script.total_duration,
        )
        return ctx


def _parse_script_response(raw: str, strategy_name: str, topic: str) -> Script:
    """Parse the JSON response from Claude into a Script."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)

    segments = [
        Segment(
            index=i,
            narration=seg["narration"],
            visual_prompt=seg.get("visual_prompt", ""),
            text_overlay=seg.get("text_overlay", ""),
        )
        for i, seg in enumerate(data["segments"])
    ]

    return Script(
        strategy_name=strategy_name,
        topic=topic,
        title=data.get("title", topic),
        segments=segments,
    )
