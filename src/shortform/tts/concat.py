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


def concat_turn_audio(
    inputs: list[Path],
    output_path: Path,
    gaps: list[float] | None = None,
    sample_rate: int = SEGMENT_SAMPLE_RATE,
    bitrate: str = SEGMENT_BITRATE,
) -> None:
    """Join `inputs` into `output_path` with `gaps[i]` seconds of silence after
    input i.

    `gaps` is padded with zeros / truncated to match `inputs`. The last entry is
    forced to zero — no trailing silence on a segment, since assembly controls
    spacing between segments and a trailing pad would desync the video mux.

    A single input still round-trips through the same filtergraph rather than
    being copied: one code path, one output profile, no "worked with 2+ turns
    but not 1" surprises.
    """
    if not inputs:
        raise ValueError("concat_turn_audio requires at least one input")

    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"concat_turn_audio: missing turn audio: {', '.join(missing)}"
        )

    padded = list(gaps or [])
    padded = (padded + [0.0] * len(inputs))[: len(inputs)]
    padded[-1] = 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"]
    for path in inputs:
        args += ["-i", str(path)]

    chains: list[str] = []
    labels: list[str] = []
    for i, gap in enumerate(padded):
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
        + f"concat=n={len(inputs)}:v=0:a=1[out]"
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

    logger.debug("Concatenating %d turns → %s", len(inputs), output_path.name)
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Turn audio concat failed for {output_path.name} "
            f"({len(inputs)} inputs): {result.stderr[-800:]}"
        )
    if not output_path.exists():
        raise RuntimeError(
            f"Turn audio concat reported success but wrote no file: {output_path}"
        )
