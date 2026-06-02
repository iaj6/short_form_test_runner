"""Tests for TTS backend failure handling.

Focus: the F5-TTS subprocess retry/timeout logic. CLAUDE.md flags the SIGSEGV
retry as load-bearing, and a hung MPS model load (no timeout) would otherwise
block the entire async pipeline forever.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from shortform.models.script import Segment
from shortform.tts import f5_backend
from shortform.tts.f5_backend import SUBPROCESS_MAX_ATTEMPTS, F5TTSBackend


def _segment() -> Segment:
    return Segment(index=0, narration="Hello there.", visual_prompt="x", text_overlay="")


def _make_backend(tmp_path: Path) -> tuple[F5TTSBackend, dict]:
    """A backend whose ref-audio + CLI existence checks pass, so synthesize()
    reaches the subprocess loop. Returns (backend, config)."""
    cli = tmp_path / "f5-cli"
    cli.write_text("#!/bin/sh\n")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    backend = F5TTSBackend(cli_path=str(cli))
    config = {"ref_audio": str(ref), "ref_text": "reference transcript"}
    return backend, config


@pytest.mark.asyncio
async def test_f5_hang_times_out_and_retries_then_fails(tmp_path: Path):
    """A subprocess that hangs (TimeoutExpired) is retried, then fails cleanly
    instead of blocking forever."""
    backend, config = _make_backend(tmp_path)

    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert "timeout" in kwargs, "subprocess.run must be called with a timeout"
        raise subprocess.TimeoutExpired(cmd="f5", timeout=kwargs["timeout"])

    with patch.object(f5_backend.subprocess, "run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="timed out"):
            await backend.synthesize(_segment(), tmp_path / "out.mp3", config)

    # Retried up to the max attempt count.
    assert calls == SUBPROCESS_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_f5_timeout_then_success(tmp_path: Path):
    """A first-attempt hang followed by a clean second attempt succeeds."""
    backend, config = _make_backend(tmp_path)

    calls = 0
    wav_name = "f5_segment_00.wav"

    def fake_run(cmd, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise subprocess.TimeoutExpired(cmd="f5", timeout=kwargs.get("timeout"))
        # Second attempt: write the expected WAV into the CLI's --output_dir.
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        (out_dir / wav_name).write_bytes(b"RIFFfakewav")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    def fake_wav_to_mp3(src, dest):
        Path(dest).write_bytes(b"mp3")

    with patch.object(f5_backend.subprocess, "run", side_effect=fake_run), \
         patch.object(f5_backend, "_wav_to_mp3", side_effect=fake_wav_to_mp3), \
         patch.object(f5_backend, "get_audio_duration", return_value=4.2):
        out = await backend.synthesize(_segment(), tmp_path / "out.mp3", config)

    assert calls == 2
    assert out.duration == 4.2
