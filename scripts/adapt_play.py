"""Chunk an adapted stage play into short-form episode scripts.

Takes a play source (data/plays/<play>/act<N>.yaml) and emits one script JSON
per episode into data/scripts/, ready for `shortform generate-from-script`.

    uv run python scripts/adapt_play.py data/plays/<play>/act1.yaml
    uv run python scripts/adapt_play.py data/plays/<play>/act1.yaml --dry-run
    uv run python scripts/adapt_play.py data/plays/<play>/act1.yaml --target 65

Source format (data/plays/ is gitignored — adapted text is content, so you supply
your own). `beat: true` forces a visual cut; `episode_break` / `episode_title`
pin an episode boundary. Speaker names are normalized, so `PERE UBU` resolves to
the `pere_ubu` key in the strategy's tts.voices:

    play: "Play Name"
    strategy: my_strategy      # a config/strategies/<name>.yaml
    act: 1
    scenes:
      - number: 1
        location: throne_room          # used to compute hero-variant keys
        staging: "A cramped painted-cardboard interior..."
        speeches:
          - speaker: FIRST CHARACTER
            line: "Their line of dialogue."
          - speaker: SECOND CHARACTER
            beat: true                 # optional: force a new visual segment
            stage_direction: "aside"   # optional: earns a longer pause, never spoken
            line: "The reply."


Two levels of chunking:

1. SPEECHES -> SEGMENTS. A segment is one visual beat, i.e. one Veo shot (with
   sub-clip chaining if it runs long). Breaks are forced at scene changes and at
   explicit `beat: true` markers in the source, and forced again once a segment
   would outrun `--segment-target`.

2. SEGMENTS -> EPISODES. Duration-bounded, but the cut point is CHOSEN rather
   than taken greedily: every candidate boundary inside the legal duration
   window is scored (scene end > explicit break > marked beat > mid-scene) and
   penalized by distance from the target, so episodes land on dramatic beats
   instead of wherever the word budget happened to run out.

Duration estimates are per-speaker, because a mixed cast doesn't have one
speaking rate: the strategy's VoiceCast says which backend voices each character,
and the engines differ by ~40% (see SPEECH_MODEL). Every constant there is fitted
against measured audio rather than guessed, and each voice's own speed setting is
applied on top — a slow blusterer and a fast schemer in the same scene otherwise
push an episode several seconds off its target.

The run also reports the VARIANT MATRIX — every distinct (location x who's on
stage) combination — which is exactly the set of reference images the visual
stage will need, and is otherwise tedious to work out by hand.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shortform.config import load_strategy  # noqa: E402
from shortform.models.script import Script, Segment, Turn  # noqa: E402
from shortform.tts.cast import (  # noqa: E402
    DEFAULT_STAGE_DIRECTION_GAP,
    DEFAULT_TURN_GAP,
    UnknownSpeakerError,
    VoiceCast,
    normalize_speaker,
)

# Speech model per backend: (words_per_second, seconds added per punctuation mark).
#
# Word count alone predicts TTS duration poorly, because engines insert prosodic
# pauses at punctuation: across three measured Ubu episodes the effective rate
# swung 2.53-2.92 w/s purely on comma density (Pere Ubu's comma-laden list of
# titles is much slower per word than plain dialogue). Splitting the model into
# rate + per-mark pause cut total error on those episodes from 8.3s to 6.1s.
#
# Both hosted rows are fitted against measured output, not guessed. Edge: 3
# episodes of one voice pair. ElevenLabs: 12 turns of the Ubu principals,
# ~0.6s mean error per turn, with each voice's `speed` setting normalised out
# so the row describes the ENGINE and rate_multiplier() handles the voice.
# Re-measure when a cast changes — probe the per-turn MP3s the TTS stage
# leaves in the working dir, then solve
#   speech_seconds = (words / wps + punctuation_marks * pause) / speed
# The F5-TTS row is NOT fitted this way — it's CLAUDE.md's ~0.55 s/word with
# prosody already baked in, hence a zero pause term.
SPEECH_MODEL: dict[str, tuple[float, float]] = {
    "edge": (3.36, 0.37),
    "elevenlabs": (2.92, 0.22),
    "f5_tts": (1.80, 0.0),
}
FALLBACK_MODEL = (3.36, 0.37)

# Marks that trigger a prosodic pause.
PAUSE_MARKS = ".!?,;:—-"

# A segment longer than this needs ~4+ chained Veo clips, where character drift
# starts to show. Flagged in the report so the adapter can add a `beat: true`.
LONG_SEGMENT_WARN = 30.0

# How hard to pull episodes toward --target. Boundary scores are 0-6, and the
# distance penalty maxes near 0.5 unweighted, so without this a marked beat would
# win at ANY legal duration and the target would be decorative.
TARGET_WEIGHT = 2.0

# Boundary desirability. Scored against distance-from-target, so a slightly
# short episode ending on a scene change beats a perfectly-sized one that cuts
# a character off mid-exchange.
SCORE_EPISODE_BREAK = 6.0  # explicit `episode_break: true` in the source
SCORE_SCENE_END = 3.0
SCORE_BEAT = 1.0
SCORE_MID_SCENE = 0.0


@dataclass
class Speech:
    """One character's speech, with its scene context resolved."""

    speaker: str
    line: str
    stage_direction: str = ""
    beat: bool = False
    episode_break: bool = False
    episode_title: str = ""
    scene_number: int = 0
    location: str = ""
    staging: str = ""
    ends_scene: bool = False
    # Estimated spoken length INCLUDING the gap that follows it; filled by
    # annotate_durations once the cast is known.
    seconds: float = 0.0

    @property
    def words(self) -> int:
        return len(self.line.split())


@dataclass
class Chunk:
    """A contiguous run of speeches — used for both segments and episodes."""

    speeches: list[Speech] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.seconds for s in self.speeches)


def _no_backends(name: str) -> Any:
    raise AssertionError(f"adapt_play does not instantiate TTS backends (asked for {name!r})")


def load_play(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not data.get("scenes"):
        raise SystemExit(f"{path}: no `scenes` found")
    return data


def flatten(play: dict[str, Any]) -> list[Speech]:
    """Walk scenes into a flat speech list carrying scene context."""
    out: list[Speech] = []
    for scene in play["scenes"]:
        speeches = scene.get("speeches") or []
        for i, raw in enumerate(speeches):
            out.append(
                Speech(
                    speaker=str(raw["speaker"]).strip(),
                    line=" ".join(str(raw["line"]).split()),
                    stage_direction=str(raw.get("stage_direction", "")).strip(),
                    beat=bool(raw.get("beat", False)),
                    episode_break=bool(raw.get("episode_break", False)),
                    episode_title=str(raw.get("episode_title", "")).strip(),
                    scene_number=int(scene.get("number", 0)),
                    location=str(scene.get("location", "")).strip(),
                    staging=" ".join(str(scene.get("staging", "")).split()),
                    ends_scene=(i == len(speeches) - 1),
                )
            )
    if not out:
        raise SystemExit("play contains no speeches")
    return out


def rate_multiplier(config: dict[str, Any]) -> float:
    """Speed multiplier from a resolved voice config.

    Backends express this differently: Edge takes a percentage string
    (`rate: "+10%"`), F5-TTS a bare float (`speed: 1.05`). Ignoring it is worth
    several seconds an episode once a cast has a slow blusterer and a fast
    schemer in it — which is exactly what character differentiation produces.
    """
    raw_rate = config.get("rate")
    if isinstance(raw_rate, str) and raw_rate.strip().endswith("%"):
        try:
            return 1.0 + float(raw_rate.strip().rstrip("%")) / 100.0
        except ValueError:
            pass

    speed = config.get("speed")
    if isinstance(speed, (int, float)) and speed > 0:
        return float(speed)

    return 1.0


def annotate_durations(
    speeches: list[Speech], cast: VoiceCast, turn_gap: float, sd_gap: float
) -> None:
    """Attach an estimated spoken duration (speech + its following gap) to each
    speech, using that speaker's backend rate scaled by their voice's speed."""
    for i, sp in enumerate(speeches):
        assignment = cast.resolve(sp.speaker)
        wps, pause = SPEECH_MODEL.get(assignment.backend_name, FALLBACK_MODEL)
        multiplier = rate_multiplier(assignment.config)
        marks = sum(sp.line.count(m) for m in PAUSE_MARKS)
        # The voice's speed setting scales the pauses too, not just the words.
        speech_seconds = (sp.words / wps + marks * pause) / multiplier

        # Gap that follows this speech: longer when the NEXT line carries a
        # stage direction, matching VoiceCast.gap_after at synthesis time.
        gap = 0.0
        if i < len(speeches) - 1:
            gap = sd_gap if speeches[i + 1].stage_direction else turn_gap
        sp.seconds = speech_seconds + gap


def boundary_score(speeches: list[Speech], idx: int) -> float:
    """How good a cut is *after* speeches[idx]."""
    here = speeches[idx]
    if here.ends_scene:
        score = SCORE_SCENE_END
    else:
        score = SCORE_MID_SCENE
    if idx + 1 < len(speeches):
        nxt = speeches[idx + 1]
        if nxt.episode_break:
            score = max(score, SCORE_EPISODE_BREAK)
        elif nxt.beat:
            score = max(score, SCORE_BEAT)
    return score


def split_episodes(
    speeches: list[Speech], target: float, min_s: float, max_s: float
) -> list[Chunk]:
    """Cut the speech stream into duration-bounded episodes at scored boundaries."""
    episodes: list[Chunk] = []
    start = 0
    while start < len(speeches):
        cum = 0.0
        best: tuple[float, int] | None = None
        last_reachable = start

        for i in range(start, len(speeches)):
            cum += speeches[i].seconds
            if cum > max_s and i > start:
                break
            last_reachable = i
            if cum >= min_s:
                # Prefer dramatic boundaries, but don't drift far from target.
                score = boundary_score(speeches, i) - TARGET_WEIGHT * abs(cum - target) / target
                if best is None or score > best[0]:
                    best = (score, i)

        cut = best[1] if best is not None else last_reachable
        episodes.append(Chunk(speeches[start : cut + 1]))
        start = cut + 1

    return episodes


def split_segments(episode: Chunk, segment_target: float) -> list[Chunk]:
    """Cut an episode into visual beats.

    Forced breaks: scene change (different backdrop) and explicit `beat: true`.
    Soft break: the running segment has reached `segment_target`.
    """
    segments: list[Chunk] = []
    current: list[Speech] = []
    running = 0.0

    for sp in episode.speeches:
        forced = bool(current) and (
            sp.beat or sp.scene_number != current[-1].scene_number
        )
        if forced or (current and running >= segment_target):
            segments.append(Chunk(current))
            current, running = [], 0.0
        current.append(sp)
        running += sp.seconds

    if current:
        segments.append(Chunk(current))
    return segments


def speakers_in(chunk: Chunk) -> list[str]:
    """Distinct speakers in order of first appearance."""
    seen: list[str] = []
    for sp in chunk.speeches:
        if sp.speaker not in seen:
            seen.append(sp.speaker)
    return seen


def variant_key(chunk: Chunk) -> str:
    """Deterministic hero-variant key: location x who's on stage.

    For an adapted play the required reference image is a pure function of these
    two facts, so VariantSelectionStage's Claude call is unnecessary here — the
    key is computed, not inferred.
    """
    location = chunk.speeches[0].location or "stage"
    who = "_".join(sorted(normalize_speaker(s) for s in speakers_in(chunk)))
    return f"{location}__{who}" if who else location


def visual_prompt(chunk: Chunk) -> str:
    """Compose a segment's visual prompt from scene staging + who's present."""
    first = chunk.speeches[0]
    who = " and ".join(speakers_in(chunk))
    parts = [first.staging] if first.staging else []
    parts.append(f"On stage: {who}.")
    directions = [
        f"{sp.speaker} {sp.stage_direction}"
        for sp in chunk.speeches
        if sp.stage_direction
    ]
    if directions:
        parts.append("Action: " + "; ".join(directions) + ".")
    return " ".join(parts)


def build_script(
    episode: Chunk,
    segments: list[Chunk],
    play: dict[str, Any],
    act: int,
    number: int,
) -> Script:
    title = next(
        (sp.episode_title for sp in episode.speeches if sp.episode_title),
        f"{play.get('play', 'Untitled')}, Part {number}",
    )
    scene_span = sorted({sp.scene_number for sp in episode.speeches})
    scene_desc = (
        f"scene {scene_span[0]}"
        if len(scene_span) == 1
        else f"scenes {scene_span[0]}-{scene_span[-1]}"
    )

    return Script(
        id=f"{_slug(play.get('play', 'play'))}{act:02d}e{number:02d}",
        strategy_name=str(play.get("strategy", "")),
        topic=f"Act {act}, {scene_desc}",
        title=title,
        segments=[
            Segment(
                index=i,
                visual_prompt=visual_prompt(seg),
                text_overlay="",
                hero_variant=variant_key(seg),
                turns=[
                    Turn(
                        speaker=sp.speaker,
                        line=sp.line,
                        stage_direction=sp.stage_direction,
                    )
                    for sp in seg.speeches
                ],
            )
            for i, seg in enumerate(segments)
        ],
    )


def _slug(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())[:8] or "play"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("play_file", type=Path, help="path to a play act YAML")
    ap.add_argument("--out", type=Path, default=Path("data/scripts"))
    ap.add_argument("--target", type=float, default=70.0, help="target episode seconds")
    ap.add_argument("--min", dest="min_s", type=float, default=45.0)
    ap.add_argument("--max", dest="max_s", type=float, default=85.0)
    ap.add_argument("--segment-target", type=float, default=22.0)
    ap.add_argument("--start-episode", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    play = load_play(args.play_file)
    act = int(play.get("act", 1))
    strategy_name = str(play.get("strategy", ""))
    if not strategy_name:
        raise SystemExit(f"{args.play_file}: no `strategy` declared")

    strategy = load_strategy(strategy_name)
    tts_cfg = strategy.tts
    cast = VoiceCast(
        base_config={},
        default_backend=str(tts_cfg.get("backend", "edge")),
        voices=tts_cfg.get("voices") or {},
        # The chunker only ever asks the cast WHICH backend voices a speaker,
        # never for an instance — no TTS runs here.
        backend_factory=_no_backends,
    )
    turn_gap = float(tts_cfg.get("turn_gap", DEFAULT_TURN_GAP))
    sd_gap = float(tts_cfg.get("stage_direction_gap", DEFAULT_STAGE_DIRECTION_GAP))

    speeches = flatten(play)

    # Fail loudly on an uncast speaker, exactly as TTSStage.validate does —
    # better to find out here than after generating 40 episode files.
    uncast = sorted({sp.speaker for sp in speeches if not cast.knows(sp.speaker)})
    if uncast:
        print(
            f"ERROR: uncast speakers in {args.play_file.name}: {', '.join(uncast)}\n"
            f"  Add them to config/strategies/{strategy_name}.yaml under tts.voices\n"
            f"  (currently cast: {', '.join(cast.speakers) or 'nobody'})",
            file=sys.stderr,
        )
        return 1

    try:
        annotate_durations(speeches, cast, turn_gap, sd_gap)
    except UnknownSpeakerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    episodes = split_episodes(speeches, args.target, args.min_s, args.max_s)

    total_words = sum(sp.words for sp in speeches)
    total_seconds = sum(ep.duration for ep in episodes)
    print(
        f"{play.get('play')} — Act {act}: {len(speeches)} speeches, "
        f"{total_words} words, {len(play['scenes'])} scene(s)"
    )
    print(
        f"-> {len(episodes)} episode(s), ~{total_seconds / 60:.1f} min total "
        f"(target {args.target:.0f}s, window {args.min_s:.0f}-{args.max_s:.0f}s)\n"
    )

    variants: dict[str, list[str]] = {}
    written: list[Path] = []

    for n, episode in enumerate(episodes, start=args.start_episode):
        segments = split_segments(episode, args.segment_target)
        script = build_script(episode, segments, play, act, n)

        for seg_chunk, seg in zip(segments, script.segments, strict=True):
            variants.setdefault(seg.hero_variant, speakers_in(seg_chunk))

        flag = "" if args.min_s <= episode.duration <= args.max_s else "  <-- OUT OF BAND"
        boundary = "scene end" if episode.speeches[-1].ends_scene else "mid-scene"
        print(
            f"  e{n:02d}  {episode.duration:5.1f}s  "
            f"{len(script.segments)} seg  {len(episode.speeches):2d} speeches  "
            f"cut@{boundary:9s}  {script.title}{flag}"
        )
        for i, (seg_chunk, seg) in enumerate(zip(segments, script.segments, strict=True)):
            warn = "  <-- long, add a `beat`" if seg_chunk.duration > LONG_SEGMENT_WARN else ""
            print(
                f"          seg {i}: {seg_chunk.duration:5.1f}s  "
                f"{len(seg.turns)} turns  [{'/'.join(speakers_in(seg_chunk))}]{warn}"
            )

        if not args.dry_run:
            out_path = args.out / f"{script.id}.json"
            script.save_json(out_path)
            written.append(out_path)

    print(f"\nVariant matrix — {len(variants)} reference image(s) needed:")
    for key in sorted(variants):
        print(f"  {key:44s} {' + '.join(variants[key])}")

    if args.dry_run:
        print("\n(dry run — nothing written)")
    else:
        print(f"\nWrote {len(written)} script(s) to {args.out}/")
        if written:
            print(f"  next: uv run shortform generate-from-script '{written[0]}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
