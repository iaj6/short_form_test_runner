"""Sound-effect library — named cues resolved from a shared manifest.

Effects are a small REUSABLE library, not per-occurrence generation. A door
slam is a door slam: generating a fresh one for every door in a 90-episode
series would cost more and sound less consistent. Same reasoning as the series
music score.

Cues are referenced by name from a play source (`sfx: door_slam`) rather than
inferred from stage-direction prose. Most stage directions are delivery notes
with no sound at all — "aside", "aloud", "low, close to his ear" — so guessing
which ones imply a noise would be wrong more often than it was right, and wrong
in a way nobody notices until they watch the episode. The author already knows.

The library is strategy-agnostic and lives at `data/sfx/`: a door slam is the
same door slam whichever series needs one.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from shortform.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SFX_DIR = PROJECT_ROOT / "data" / "sfx"
MANIFEST_NAME = "library.yaml"


@dataclass(frozen=True)
class Effect:
    name: str
    path: Path
    description: str = ""


@functools.lru_cache(maxsize=1)
def load_library(sfx_dir: Path | None = None) -> dict[str, Effect]:
    """Effect name -> Effect, for cues whose audio is actually on disk.

    Cached: this is read once per turn during TTS, and the manifest doesn't
    change mid-run. Entries with no audio file are skipped rather than raising —
    the manifest is committed while the audio is gitignored, so a fresh clone
    legitimately has cues with nothing behind them yet.
    """
    directory = sfx_dir or SFX_DIR
    manifest = directory / MANIFEST_NAME
    if not manifest.exists():
        return {}

    try:
        data = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError as e:
        logger.warning("Unreadable sfx manifest %s (%s) — no effects available", manifest, e)
        return {}

    library: dict[str, Effect] = {}
    for raw in data.get("effects") or []:
        name = str(raw.get("name", "")).strip()
        path = directory / str(raw.get("file", ""))
        if not name:
            continue
        if not path.exists():
            logger.warning(
                "sfx '%s' is declared but %s is missing — skipping "
                "(audio is gitignored; run scripts/generate_sfx.py)",
                name, path.name,
            )
            continue
        library[name] = Effect(
            name=name, path=path, description=str(raw.get("description", ""))
        )
    return library


def resolve(name: str, sfx_dir: Path | None = None) -> Path | None:
    """Audio path for a cue name, or None if it isn't available.

    An unknown cue is a warning rather than an error. A missing sound effect
    degrades an episode; failing the render throws away everything already
    generated for it, which is much worse.
    """
    if not name:
        return None
    library = load_library(sfx_dir)
    effect = library.get(name)
    if effect is None:
        known = ", ".join(sorted(library)) or "none loaded"
        logger.warning("Unknown sfx cue '%s' — skipping it (available: %s)", name, known)
        return None
    return effect.path
