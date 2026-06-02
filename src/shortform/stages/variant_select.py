"""Variant selection stage — picks per-segment hero reference variants.

Sits between script_gen and tts. For strategies that declare a hero-variant
manifest (see `data/character_refs/variants.yaml`), this stage asks Claude
to pick the most appropriate variant for each segment based on its
narration + visual_prompt. The chosen `hero_variant` key gets written to
each Segment; VisualGenStage uses it to resolve the per-segment reference
image before calling the backend.

Strategies without a `visuals.variants_manifest` field are a no-op for
this stage — it just returns the context unchanged. So the stage is safe
to keep in the pipeline for all strategies, not just gothic_vignette.
"""

from __future__ import annotations

import logging
from typing import Any

import anthropic
import yaml

from shortform.config import PROJECT_ROOT
from shortform.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You select the best hero-reference variant for each segment of a short-form \
video script. The character is fixed (same wardrobe, same face, same \
proportions across every variant) — the variants differ only in the SCENE \
the character is placed in (e.g., at a desk, in a kitchen, by a window).

The goal of variant selection is to give the downstream image-to-video model \
a starting frame whose scene already MATCHES what each segment's narration \
is about, so the model doesn't have to morph an unrelated environment to fit \
the script. A kitchen scene for a dishwasher vignette; a laptop scene for an \
email vignette; a window scene for a melancholic-contemplation vignette.

Rules:
- Coherence within a video usually beats novelty. If two adjacent segments \
  can reasonably share a variant, prefer that — fewer scene cuts read more \
  intentional. But don't force coherence: a vignette that legitimately moves \
  between scenes (e.g., kitchen → study → window) should pick variants that \
  match each scene.
- Use the segment's `visual_prompt` and `narration` to judge which scene fits.
- If a segment doesn't strongly suggest any particular variant, default to \
  the same variant the previous segment used, or `study` if it's the first \
  segment.
- Brief one-sentence `reason` per selection so the choice is auditable later.
"""


class VariantSelectionStage:
    @property
    def name(self) -> str:
        return "variant_select"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.script.segments:
            errors.append("No script segments for variant selection")
        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        manifest_rel = ctx.strategy.visuals.get("variants_manifest")
        if not manifest_rel:
            logger.info(
                "Strategy '%s' has no variants_manifest — skipping variant selection",
                ctx.strategy.name,
            )
            return ctx

        manifest_path = PROJECT_ROOT / manifest_rel
        if not manifest_path.exists():
            logger.warning(
                "variants_manifest %s not found — skipping variant selection",
                manifest_path,
            )
            return ctx

        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        variants = manifest.get("variants", [])
        if not variants:
            logger.warning("variants_manifest %s has no variants — skipping", manifest_path)
            return ctx

        valid_keys = {v["key"] for v in variants}
        default_key = ctx.strategy.visuals.get("default_variant") or next(iter(valid_keys))

        # If every segment already has hero_variant set (e.g., from a manually-
        # edited script JSON), respect that and bail.
        if all(seg.hero_variant for seg in ctx.script.segments):
            logger.info(
                "All %d segments already have hero_variant set — preserving",
                len(ctx.script.segments),
            )
            return ctx

        if not ctx.settings.anthropic_api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not configured — defaulting all segments to '%s'",
                default_key,
            )
            for seg in ctx.script.segments:
                if not seg.hero_variant:
                    seg.hero_variant = default_key
            return ctx

        selections = _ask_claude_to_pick(
            api_key=ctx.settings.anthropic_api_key,
            model=ctx.settings.llm.model,
            variants=variants,
            segments=ctx.script.segments,
        )

        # Apply selections to segments; fall back to default for missing/invalid
        chosen_by_idx: dict[int, dict[str, str]] = {
            int(s["segment_index"]): s for s in selections
        }
        for seg in ctx.script.segments:
            if seg.hero_variant:
                continue
            pick = chosen_by_idx.get(seg.index)
            key = (pick or {}).get("variant_key", "")
            if key not in valid_keys:
                logger.warning(
                    "Segment %d: invalid/missing variant '%s' from Claude, defaulting to '%s'",
                    seg.index, key, default_key,
                )
                key = default_key
            seg.hero_variant = key
            reason = (pick or {}).get("reason", "")
            logger.info(
                "Segment %d → variant '%s'  (%s)",
                seg.index, seg.hero_variant, reason or "default",
            )

        return ctx


def _ask_claude_to_pick(
    api_key: str,
    model: str,
    variants: list[dict[str, Any]],
    segments: list[Any],
) -> list[dict[str, str]]:
    """Tool-use call: Claude picks one variant per segment."""
    variant_summary = "\n\n".join(
        f"key: {v['key']}\n"
        f"fits: {v.get('fits', '').strip()}\n"
        f"scene: {v.get('description', '').strip()}"
        for v in variants
    )
    segment_summary = "\n\n".join(
        f"[segment {seg.index}]\n"
        f"narration: {seg.narration}\n"
        f"visual_prompt: {seg.visual_prompt}"
        for seg in segments
    )

    user_msg = (
        f"AVAILABLE VARIANTS:\n\n{variant_summary}\n\n"
        f"---\n\nSCRIPT SEGMENTS:\n\n{segment_summary}\n\n"
        f"---\n\n"
        f"Pick one variant_key per segment. Return one entry per segment, "
        f"in segment_index order."
    )

    tool: dict[str, Any] = {
        "name": "record_variant_selections",
        "description": "Record the chosen variant for each segment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "segment_index": {"type": "integer"},
                            "variant_key": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["segment_index", "variant_key", "reason"],
                    },
                }
            },
            "required": ["selections"],
        },
    }

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_variant_selections"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_variant_selections":
            return list(block.input["selections"])  # type: ignore[index]

    raise RuntimeError(
        "Claude did not call record_variant_selections. Raw response: "
        f"{response.model_dump_json()[:400]}"
    )
