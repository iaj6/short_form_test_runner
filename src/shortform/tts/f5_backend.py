"""F5-TTS backend — voice-cloned TTS via the f5-tts CLI in a separate venv.

Architecture: F5-TTS lives in its own Python environment outside the project
(torch + model weights are ~3GB; keeping them out of the project venv lets
people clone the repo without paying that cost if they only use Edge TTS).
This backend subprocesses `f5-tts_infer-cli` from that external venv.

Trade-off: every invocation pays the model-load cost on cold start (~3 min
on first run, faster on subsequent runs as weights are filesystem-cached).
For production batch generation we'll want to swap to a persistent service
(FastAPI on localhost), but for first integration this is the cleanest path.

Strategy YAML must provide:
  tts:
    backend: "f5_tts"
    ref_audio: "data/voices/bartholomew_reference_trimmed.wav"
    ref_text: "Good evening. Tonight, I shall speak slowly..."
    model: "F5TTS_v1_Base"     # optional
    speed: 1.0                 # optional
    cfg_strength: 2.0          # optional

F5-TTS does not provide per-word timings, so word_timings on the returned
TTSOutput is empty. Strategies that depend on animated subtitles should
stick with Edge TTS until we wire up Whisper for post-hoc alignment.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from shortform.config import PROJECT_ROOT
from shortform.models.script import Segment
from shortform.tts.backend import TTSOutput, get_audio_duration

logger = logging.getLogger(__name__)

DEFAULT_CLI_PATH = "~/.venvs/f5-tts/bin/f5-tts_infer-cli"
DEFAULT_MODEL = "F5TTS_v1_Base"

# F5-TTS occasionally segfaults at MPS model load on Apple Silicon (exit -11).
# Pure transient — retrying with the same input usually succeeds. Other
# non-zero exits (positive codes) are also worth a retry since memory/disk
# pressure mid-load can cause them and they similarly clear on a second try.
SUBPROCESS_MAX_ATTEMPTS = 2

# Hard wall-clock ceiling per inference invocation. Cold start (model load) can
# legitimately take ~3 min, but under memory/disk pressure the MPS load can
# *hang* rather than segfault — and subprocess.run with no timeout would block
# the entire async pipeline forever with no recovery. 10 min is generous
# headroom over a healthy cold start; a hang is then treated as a failed
# attempt (retried once, then raised cleanly) like any other transient failure.
SUBPROCESS_TIMEOUT_SECONDS = 600


class F5TTSBackend:
    """F5-TTS via subprocess to its dedicated venv."""

    def __init__(self, cli_path: str = DEFAULT_CLI_PATH) -> None:
        self.cli_path = str(Path(cli_path).expanduser())

    @property
    def name(self) -> str:
        return "f5_tts"

    async def synthesize(
        self,
        segment: Segment,
        output_path: Path,
        config: dict[str, Any],
    ) -> TTSOutput:
        ref_audio = config.get("ref_audio")
        ref_text = config.get("ref_text")
        if not ref_audio:
            raise RuntimeError(
                "F5-TTS backend requires 'ref_audio' in strategy.tts config"
            )
        if not ref_text:
            raise RuntimeError(
                "F5-TTS backend requires 'ref_text' in strategy.tts config"
            )

        ref_audio_path = Path(ref_audio)
        if not ref_audio_path.is_absolute():
            ref_audio_path = PROJECT_ROOT / ref_audio
        if not ref_audio_path.exists():
            raise RuntimeError(f"F5-TTS ref_audio not found: {ref_audio_path}")

        if not Path(self.cli_path).exists():
            raise RuntimeError(
                f"F5-TTS CLI not found at {self.cli_path}. "
                "Install with: uv venv ~/.venvs/f5-tts --python 3.12 && "
                "uv pip install --python ~/.venvs/f5-tts/bin/python f5-tts"
            )

        model = config.get("model", DEFAULT_MODEL)
        speed = config.get("speed", 1.0)
        cfg_strength = config.get("cfg_strength", 2.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_filename = f"f5_segment_{segment.index:02d}.wav"
            cmd = [
                self.cli_path,
                "--model", model,
                "--ref_audio", str(ref_audio_path),
                "--ref_text", ref_text,
                "--gen_text", segment.narration,
                "--output_dir", tmpdir,
                "--output_file", wav_filename,
                "--speed", str(speed),
                "--cfg_strength", str(cfg_strength),
            ]
            logger.info(
                "F5-TTS segment %d (%d chars, model=%s)",
                segment.index,
                len(segment.narration),
                model,
            )
            result: subprocess.CompletedProcess[str] | None = None
            for attempt in range(1, SUBPROCESS_MAX_ATTEMPTS + 1):
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    # A hang (not just a non-zero exit) — treat like a transient
                    # failure: retry once, then fail cleanly instead of blocking
                    # the pipeline forever.
                    logger.warning(
                        "F5-TTS segment %d timed out after %ds (attempt %d/%d), retrying...",
                        segment.index, SUBPROCESS_TIMEOUT_SECONDS, attempt,
                        SUBPROCESS_MAX_ATTEMPTS,
                    )
                    result = None
                    if attempt < SUBPROCESS_MAX_ATTEMPTS:
                        continue
                    raise RuntimeError(
                        f"f5-tts_infer-cli timed out after {SUBPROCESS_MAX_ATTEMPTS} "
                        f"attempts ({SUBPROCESS_TIMEOUT_SECONDS}s each) on segment "
                        f"{segment.index}"
                    ) from None
                if result.returncode == 0:
                    break
                if attempt < SUBPROCESS_MAX_ATTEMPTS:
                    sig_hint = (
                        " (SIGSEGV — MPS model-load crash, common on Apple Silicon)"
                        if result.returncode == -11 else ""
                    )
                    logger.warning(
                        "F5-TTS segment %d failed (exit %d, attempt %d/%d)%s, retrying...",
                        segment.index, result.returncode, attempt,
                        SUBPROCESS_MAX_ATTEMPTS, sig_hint,
                    )

            assert result is not None  # loop always runs at least once
            if result.returncode != 0:
                raise RuntimeError(
                    f"f5-tts_infer-cli failed after {SUBPROCESS_MAX_ATTEMPTS} "
                    f"attempts (final exit {result.returncode}):\n"
                    f"stderr (tail): {result.stderr[-800:]}"
                )

            wav_path = Path(tmpdir) / wav_filename
            if not wav_path.exists():
                raise RuntimeError(
                    f"f5-tts_infer-cli reported success but no output at "
                    f"{wav_path}.\nstdout (tail): {result.stdout[-500:]}"
                )

            _wav_to_mp3(wav_path, output_path)

        duration = get_audio_duration(output_path)
        return TTSOutput(
            audio_path=output_path,
            duration=duration,
            word_timings=[],
        )


def _wav_to_mp3(src: Path, dest: Path) -> None:
    """Convert F5-TTS's 24kHz WAV output to MP3 to match file_store's .mp3 contract."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-c:a", "libmp3lame", "-b:a", "96k",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg WAV→MP3 conversion failed: {result.stderr}")
