"""Concatenate per-turn audio into one segment track.

Dialogue segments synthesize each turn separately, possibly through different
backends, then join them here. The inputs are therefore heterogeneous: F5-TTS
emits 24kHz mono MP3 via its WAV conversion, Edge TTS emits its own MP3 profile,
and a future backend could differ again.

That rules out ffmpeg's concat *demuxer* — it requires identical codec
parameters across inputs and produces garbage (or a hard failure) when they
differ. We use the concat *filter* with per-input aresample/aformat
normalization, which is the same lesson assembly.py already encodes for video
xfade chains where Veo's varying timebases tripped the filter.

Output matches the profile f5_backend._wav_to_mp3 writes, so a concatenated
multi-voice segment is indistinguishable from a single-voice one to every
downstream stage — assembly reads `seg.audio_path` and nothing else.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Matches settings.video.audio_sample_rate, so hosted backends that return
# 44.1kHz (ElevenLabs) aren't downsampled here and resampled back up in the
# final master. Local backends emit 24kHz; upsampling those early is a no-op.
SEGMENT_SAMPLE_RATE = 44100
SEGMENT_BITRATE = "128k"

# Beat between the end of a line and a sound effect that follows it, so the
# effect doesn't clip the tail of the speech.
SFX_LEAD_GAP = 0.15


def build_clip_plan(
    inputs: list[Path],
    gaps: list[float] | None = None,
    gap_audio: list[Path | None] | None = None,
) -> list[tuple[Path, float]]:
    """Ordered (audio, silence-after) pairs for one segment.

    Turns and sound effects are the same kind of thing here — a file with a pad
    after it — so an effect is simply another clip inserted after the line it
    annotates:

        line  ->  SFX_LEAD_GAP  ->  effect  ->  gap  ->  next line

    Returned rather than used directly so callers can compute where each turn
    LANDS without re-deriving the layout. The video stage tells the model who
    speaks when using those offsets, and an effect inserted mid-segment shifts
    every turn after it — computing the plan once means the audio and the
    speech schedule can't disagree.

    The final pad is forced to zero: assembly controls spacing between segments,
    and a trailing pad desyncs the video mux.
    """
    padded = list(gaps or [])
    padded = (padded + [0.0] * len(inputs))[: len(inputs)]
    effects = list(gap_audio or [])
    effects = (effects + [None] * len(inputs))[: len(inputs)]

    plan: list[tuple[Path, float]] = []
    for turn_audio, gap, effect in zip(inputs, padded, effects, strict=True):
        if effect is not None:
            plan.append((turn_audio, SFX_LEAD_GAP))
            plan.append((effect, gap))
        else:
            plan.append((turn_audio, gap))

    if plan:
        plan[-1] = (plan[-1][0], 0.0)
    return plan


def concat_turn_audio(
    inputs: list[Path],
    output_path: Path,
    gaps: list[float] | None = None,
    sample_rate: int = SEGMENT_SAMPLE_RATE,
    bitrate: str = SEGMENT_BITRATE,
    gap_audio: list[Path | None] | None = None,
) -> None:
    """Join `inputs` into `output_path` with `gaps[i]` seconds of silence after
    input i.

    `gap_audio[i]`, when set, places a sound effect in that gap instead of plain
    silence — the pause a stage direction already earns is exactly where its
    effect belongs.

    A single input still round-trips through the same filtergraph rather than
    being copied: one code path, one output profile, no "worked with 2+ turns
    but not 1" surprises.
    """
    if not inputs:
        raise ValueError("concat_turn_audio requires at least one input")

    plan = build_clip_plan(inputs, gaps, gap_audio)
    missing = [str(path) for path, _ in plan if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"concat_turn_audio: missing audio: {', '.join(missing)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    for path, _ in plan:
        args += ["-i", str(path)]

    chains: list[str] = []
    labels: list[str] = []
    for i, (_, gap) in enumerate(plan):
        chain = (
            f"[{i}:a]aresample={sample_rate}"
            f",aformat=sample_fmts=fltp:channel_layouts=mono"
            f",asetpts=PTS-STARTPTS"
        )
        if gap > 0:
            # pad_dur appends exactly this much silence and terminates; bare
            # apad would pad forever and hang the concat.
            chain += f",apad=pad_dur={gap:.3f}"
        chains.append(f"{chain}[a{i}]")
        labels.append(f"[a{i}]")

    filtergraph = (
        ";".join(chains)
        + ";"
        + "".join(labels)
        + f"concat=n={len(plan)}:v=0:a=1[out]"
    )

    args += [
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]

    logger.debug(
        "Concatenating %d clip(s) from %d turn(s) → %s",
        len(plan), len(inputs), output_path.name,
    )
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Turn audio concat failed for {output_path.name} "
            f"({len(plan)} clips): {result.stderr[-800:]}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"Turn audio concat reported success but wrote no file: {output_path}"
        )
