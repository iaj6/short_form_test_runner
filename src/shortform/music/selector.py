"""Music track selection from a curated `tracks.yaml` manifest.

Assembly previously picked a track with `random.choice` over a directory glob.
The mood/tempo/intensity metadata curated in `data/music/<category>/tracks.yaml`
was documented as advisory and read by nothing.

Two problems with random, only one of which is obvious:

1. A track that fits the content beats one that doesn't. That's the visible win.
2. **Random selection is not stable across re-renders.** The pipeline resumes
   interrupted runs and re-renders episodes; with `random.choice` the same
   episode gets different music every time it's rebuilt, so a batch you resumed
   halfway through ends up scored inconsistently. Selection here is
   deterministic in the episode's seed (its script id): the same episode always
   resolves to the same track, while different episodes still spread across the
   library.

Matching is a small weighted score rather than a filter, so a partial match
still returns something — a strategy asking for moods no track carries gets the
closest available rather than silence.

Categories without a manifest (`ambient/`, `upbeat/`) fall back to a directory
glob, still seeded, so nothing regresses for strategies that never curated one.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MANIFEST_NAME = "tracks.yaml"
AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".m4a", ".ogg")

# Each shared mood tag is worth more than a tempo or intensity match: mood is
# what makes a cue feel right under a scene, while tempo/intensity are
# tie-breakers that keep a frantic track off a contemplative beat.
MOOD_WEIGHT = 3.0
TEMPO_WEIGHT = 2.0
INTENSITY_WEIGHT = 2.0


@dataclass
class Track:
    """One entry from a category manifest."""

    path: Path
    title: str = ""
    composer: str = ""
    license: str = ""
    attribution_required: bool = False
    attribution_text: str = ""
    mood: list[str] = field(default_factory=list)
    tempo: str = ""
    intensity: str = ""
    loopable: bool = True

    @property
    def credit(self) -> str:
        """Attribution line for a video description, or "" if none is needed."""
        if not self.attribution_required:
            return ""
        if self.attribution_text:
            return self.attribution_text
        # A track flagged as needing credit but carrying no text is a curation
        # gap; emit something usable rather than silently crediting no one.
        parts = [p for p in (self.title, self.composer) if p]
        return " — ".join(parts) if parts else self.path.name


@dataclass
class MusicRequest:
    """What the content wants from its score."""

    mood: list[str] = field(default_factory=list)
    tempo: str = ""
    intensity: str = ""
    # Stable per-episode string (the script id). Same seed -> same track, so a
    # re-rendered or resumed episode keeps the music it had.
    seed: str = ""


def load_tracks(category_dir: Path) -> list[Track]:
    """Tracks from `<category>/tracks.yaml`, or [] when there's no manifest.

    Entries whose audio file is missing are skipped with a warning — the
    manifest is checked into git while the audio is gitignored, so a fresh
    clone legitimately has entries with no file behind them.
    """
    manifest = category_dir / MANIFEST_NAME
    if not manifest.exists():
        return []

    try:
        data = yaml.safe_load(manifest.read_text()) or {}
    except yaml.YAMLError as e:
        logger.warning("Unreadable %s (%s) — falling back to directory glob", manifest, e)
        return []

    tracks: list[Track] = []
    for raw in data.get("tracks") or []:
        path = category_dir / str(raw.get("file", ""))
        if not path.exists():
            logger.warning(
                "Manifest lists %s but the file is missing — skipping "
                "(audio is gitignored; re-download it)", path.name,
            )
            continue
        tracks.append(
            Track(
                path=path,
                title=str(raw.get("title", "")),
                composer=str(raw.get("composer", "")),
                license=str(raw.get("license", "")),
                attribution_required=bool(raw.get("attribution_required", False)),
                attribution_text=str(raw.get("attribution_text", "")),
                mood=[str(m) for m in raw.get("mood") or []],
                tempo=str(raw.get("tempo", "")),
                intensity=str(raw.get("intensity", "")),
                loopable=bool(raw.get("loopable", True)),
            )
        )
    return tracks


def score_track(track: Track, request: MusicRequest) -> float:
    """How well `track` fits `request`. Higher is better; 0.0 means no signal."""
    wanted = {m.lower() for m in request.mood}
    have = {m.lower() for m in track.mood}
    score = MOOD_WEIGHT * len(wanted & have)

    if request.tempo and track.tempo and request.tempo.lower() == track.tempo.lower():
        score += TEMPO_WEIGHT
    if (
        request.intensity
        and track.intensity
        and request.intensity.lower() == track.intensity.lower()
    ):
        score += INTENSITY_WEIGHT
    return score


def _seeded_index(seed: str, count: int) -> int:
    """Stable index in [0, count) derived from `seed`.

    Deliberately not `random` — the point is that the same episode resolves to
    the same track on every re-render, while different episodes spread out.
    """
    if count <= 1:
        return 0
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % count


def select_track(
    category_dir: Path, request: MusicRequest | None = None
) -> Track | None:
    """Best-fitting track for `request`, or None when the category is empty.

    Ties are broken by seed rather than by manifest order, so a strategy whose
    moods match several tracks equally still spreads across them instead of
    always taking the first.
    """
    request = request or MusicRequest()
    tracks = load_tracks(category_dir)

    if not tracks:
        # No manifest (or every entry's file is missing) — glob the directory so
        # uncurated categories keep working.
        files = sorted(
            p for p in category_dir.glob("*") if p.suffix.lower() in AUDIO_SUFFIXES
        )
        if not files:
            return None
        glob_pick = files[_seeded_index(request.seed, len(files))]
        logger.info(
            "Music: %s (no manifest in %s — picked by directory glob)",
            glob_pick.name, category_dir.name,
        )
        return Track(path=glob_pick)

    scored = [(score_track(t, request), t) for t in tracks]
    best = max(s for s, _ in scored)
    candidates = [t for s, t in scored if s == best]
    chosen = candidates[_seeded_index(request.seed, len(candidates))]

    if best <= 0 and request.mood:
        logger.info(
            "Music: %s (no track matched mood %s in '%s' — using seeded pick "
            "from %d track(s))",
            chosen.path.name, request.mood, category_dir.name, len(candidates),
        )
    else:
        logger.info(
            "Music: %s (score %.0f, %d of %d track(s) tied)",
            chosen.path.name, best, len(candidates), len(tracks),
        )
    return chosen
