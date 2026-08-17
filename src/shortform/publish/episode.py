"""Resolve a rendered episode into something uploadable.

The pipeline already writes everything an upload needs, just scattered: the
video and its `<id>.manifest.json` sidecar in data/videos/, the script JSON in
data/scripts/, and the channel's voice in the strategy YAML. This module is the
join, kept out of the CLI so the metadata rules are testable without a network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shortform.config import PROJECT_ROOT, load_strategy

logger = logging.getLogger(__name__)

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"
SCRIPT_DIR = PROJECT_ROOT / "data" / "scripts"

# Vertical and under three minutes qualifies as a Short; the hashtag is the
# conventional signal that the format is intended. Cheap to include and the
# only thing separating a Short from an ordinary vertical upload.
SHORTS_TAG = "#Shorts"


@dataclass
class Episode:
    """A rendered episode and everything needed to describe it."""

    video_id: str
    video_path: Path
    title: str
    strategy_name: str
    topic: str = ""
    flagged: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    publish_config: dict[str, Any] = field(default_factory=dict)


def load_episode(video_id: str) -> Episode:
    """Gather an episode's video, manifest and script.

    The manifest is required rather than optional: it is what records that a
    render actually finished, and publishing a video with no manifest would
    mean uploading something whose provenance the pipeline can't vouch for.
    """
    manifest_path = VIDEO_DIR / f"{video_id}.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest_path}. Only episodes rendered by "
            "`shortform batch` carry one — `generate-from-script` does not."
        )

    manifest = json.loads(manifest_path.read_text())
    video_path = VIDEO_DIR / manifest.get("output", "")
    if not video_path.exists():
        raise FileNotFoundError(
            f"manifest names {manifest.get('output')!r} but that file is missing"
        )

    strategy_name, topic = "", ""
    script_path = SCRIPT_DIR / f"{video_id}.json"
    if script_path.exists():
        script = json.loads(script_path.read_text())
        strategy_name = script.get("strategy_name", "")
        topic = script.get("topic", "")
    else:
        logger.warning(
            "No script at %s — falling back to manifest metadata only", script_path
        )

    publish_config: dict[str, Any] = {}
    if strategy_name:
        try:
            publish_config = load_strategy(strategy_name).publish
        except Exception as e:  # noqa: BLE001 — a missing strategy must not block
            logger.warning("Could not load strategy %r: %s", strategy_name, e)

    return Episode(
        video_id=video_id,
        video_path=video_path,
        title=manifest.get("title", video_id),
        strategy_name=strategy_name,
        topic=topic,
        flagged=list(manifest.get("flagged") or []),
        unverified=list(manifest.get("unverified") or []),
        publish_config=publish_config,
    )


def build_description(ep: Episode) -> str:
    """The upload description.

    A strategy supplies `publish.description`, which may reference {title} and
    {topic}. Everything else is appended rather than substituted, so a channel
    can't accidentally drop the Shorts tag by overriding the template.
    """
    template = str(ep.publish_config.get("description", "")).strip()
    if template:
        body = template.format(title=ep.title, topic=ep.topic)
    elif ep.topic:
        body = f"{ep.title} — {ep.topic}"
    else:
        body = ep.title

    parts = [body]
    footer = str(ep.publish_config.get("footer", "")).strip()
    if footer:
        parts.append(footer)
    if SHORTS_TAG not in body:
        parts.append(SHORTS_TAG)
    return "\n\n".join(parts)


def build_tags(ep: Episode) -> list[str]:
    """Strategy tags, deduplicated, preserving order.

    YouTube counts tags against a 500-character budget across all of them, so
    a long list silently costs the later entries; the strategy's own ordering
    decides what survives.
    """
    seen: list[str] = []
    for tag in ep.publish_config.get("tags", []) or []:
        text = str(tag).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def blocking_reasons(ep: Episode, allow_flagged: bool = False) -> list[str]:
    """Why this episode should not be uploaded yet.

    Flagged clips block by default. The critic flags a clip only after
    exhausting its regenerate ladder, so a flag means a human was supposed to
    look — and an upload costs ~1600 quota units out of 10,000/day, which is
    roughly six per day. Spending one on a known-bad episode is the waste
    worth preventing, quite apart from what lands on the channel.

    `unverified` deliberately does NOT block: "never checked" is a weaker
    signal than "checked and failed", and blocking on it would stop every
    upload made without an ANTHROPIC_API_KEY.
    """
    reasons = []
    if ep.flagged and not allow_flagged:
        reasons.append(
            f"{len(ep.flagged)} clip(s) flagged by the continuity critic: "
            f"{', '.join(ep.flagged)}. Watch them, then pass --allow-flagged."
        )
    return reasons
