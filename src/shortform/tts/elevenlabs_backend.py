"""ElevenLabs TTS backend — hosted, expressive, character-capable voices.

Why this exists alongside F5-TTS. CLAUDE.md records "local F5-TTS over hosted
ElevenLabs — free, high quality, no character limits" as a settled decision, and
for NARRATION it still holds: one voice, unbounded volume, per-character billing
is the enemy. An adapted stage play inverts both premises — a play needs ~20
distinct character voices and its total volume is bounded (Ubu roi is ~85k
characters of speech in total). Recording a reference clip per character doesn't
scale, and Edge's catalogue is uniformly "friendly, positive" with no grotesques
in it. So dialogue strategies use this; Bartholomew keeps F5.

Uses the `/with-timestamps` endpoint rather than plain synthesis: it returns
character-level alignment alongside the audio, which we fold into word timings.
That means burned captions come free and exact, with no Whisper transcription
pass — faster than the F5 path and more accurate, since it's real alignment
rather than ASR guessing at what was said.

Strategy YAML:

    tts:
      backend: "elevenlabs"
      voices:
        pere_ubu: { voice_id: "N2lVS1w4EtoT3dr4eOWO", stability: 0.35 }
        mere_ubu: { voice_id: "pFZP5JQG7iQjIQuC4Bku" }
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from shortform.models.script import Segment, WordTiming
from shortform.tts.backend import TTSOutput, get_audio_duration

logger = logging.getLogger(__name__)

API_ROOT = "https://api.elevenlabs.io/v1"
DEFAULT_MODEL = "eleven_multilingual_v2"

# 44.1kHz matches the final master's sample rate (settings.video), so nothing
# downsamples the hosted audio on its way through concat + assembly.
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

# Transient failures worth retrying: rate limits and upstream 5xx. Authentication
# and permission errors are NOT retried — a key missing `text_to_speech` will
# never succeed, and retrying just delays a clear error message.
MAX_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 120.0

# voice_settings keys we forward when a strategy sets them. Anything else in the
# merged config (ref_audio, rate, ...) belongs to another backend and is ignored.
_VOICE_SETTING_KEYS = ("stability", "similarity_boost", "style", "use_speaker_boost", "speed")


class ElevenLabsBackend:
    """Hosted TTS via the ElevenLabs REST API."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "elevenlabs"

    async def synthesize(
        self,
        segment: Segment,
        output_path: Path,
        config: dict[str, Any],
    ) -> TTSOutput:
        if not self.api_key:
            raise RuntimeError(
                "ElevenLabs backend requires an API key. Set ELEVENLABS_API_KEY "
                "in your .env file."
            )
        voice_id = config.get("voice_id")
        if not voice_id:
            raise RuntimeError(
                "ElevenLabs backend requires 'voice_id'. Set it per speaker under "
                "strategy.tts.voices, or globally under strategy.tts."
            )

        payload: dict[str, Any] = {
            "text": segment.narration,
            "model_id": config.get("model_id", DEFAULT_MODEL),
        }
        voice_settings = {
            k: config[k] for k in _VOICE_SETTING_KEYS if config.get(k) is not None
        }
        if voice_settings:
            payload["voice_settings"] = voice_settings

        # Pronunciation dictionary, applied server-side at synthesis. Needed when
        # written and spoken forms must differ: "Pere Ubu" has to SOUND like
        # "Pair Ooboo" but still READ as "Pere Ubu" in the burned captions.
        # Rewriting the source text instead would fix the audio and break the
        # captions.
        locator = _dictionary_locator(config)
        if locator:
            payload["pronunciation_dictionary_locators"] = [locator]

        url = f"{API_ROOT}/text-to-speech/{voice_id}/with-timestamps"
        params = {"output_format": config.get("output_format", DEFAULT_OUTPUT_FORMAT)}

        logger.info(
            "ElevenLabs segment %d (%d chars, voice=%s)",
            segment.index, len(segment.narration), voice_id,
        )
        data = await self._post_with_retry(url, payload, params)

        audio_b64 = data.get("audio_base64")
        if not audio_b64:
            raise RuntimeError(
                f"ElevenLabs returned no audio for segment {segment.index}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(audio_b64))

        # Use `alignment`, NOT `normalized_alignment`. Verified against the API:
        # with a pronunciation dictionary active, normalized_alignment carries the
        # text as SPOKEN — i.e. the aliases — so captions would read "Pair Ooboo"
        # instead of "Pere Ubu". `alignment` keeps the original text while still
        # timing it against the aliased audio, which is exactly what a burned
        # caption needs. The cost is that numbers stay as digits rather than being
        # spelled out, which is fine (arguably better) on screen.
        alignment = data.get("alignment") or data.get("normalized_alignment") or {}
        return TTSOutput(
            audio_path=output_path,
            duration=get_audio_duration(output_path),
            word_timings=alignment_to_word_timings(alignment),
        )

    async def _post_with_retry(
        self, url: str, payload: dict[str, Any], params: dict[str, str]
    ) -> dict[str, Any]:
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        last_error = ""

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = await client.post(
                        url, json=payload, params=params, headers=headers
                    )
                except httpx.RequestError as e:  # network-level, worth retrying
                    last_error = f"request failed: {e}"
                else:
                    if response.status_code == 200:
                        result: dict[str, Any] = response.json()
                        return result
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    # 401/403 are configuration errors — fail fast with the
                    # server's message rather than burning three attempts.
                    if response.status_code in (401, 403):
                        raise RuntimeError(f"ElevenLabs rejected the request — {last_error}")
                    if response.status_code < 500 and response.status_code != 429:
                        raise RuntimeError(f"ElevenLabs request failed — {last_error}")

                if attempt < MAX_ATTEMPTS:
                    delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning(
                        "ElevenLabs attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt, MAX_ATTEMPTS, last_error, delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"ElevenLabs failed after {MAX_ATTEMPTS} attempts — {last_error}"
        )


def _dictionary_locator(config: dict[str, Any]) -> dict[str, str] | None:
    """Build the pronunciation-dictionary locator from strategy config.

    Both an id and a version are required — the API pins a specific version, so
    editing a dictionary in the dashboard mints a new version_id and existing
    strategies keep using the old one until updated. That is deliberate: a
    surprise pronunciation change partway through a series would be worse than
    a stale one.
    """
    dict_id = config.get("pronunciation_dictionary_id")
    version_id = config.get("pronunciation_dictionary_version_id")
    if not dict_id or not version_id:
        return None
    return {
        "pronunciation_dictionary_id": str(dict_id),
        "version_id": str(version_id),
    }


def alignment_to_word_timings(alignment: dict[str, Any]) -> list[WordTiming]:
    """Fold ElevenLabs' per-CHARACTER alignment into per-word timings.

    The API reports a start/end for every character including spaces, so words
    are recovered by accumulating runs of non-whitespace: a word starts at its
    first character's start and ends at its last character's end.
    """
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    if not chars or len(chars) != len(starts) or len(chars) != len(ends):
        return []

    timings: list[WordTiming] = []
    word = ""
    word_start = 0.0
    word_end = 0.0

    for char, start, end in zip(chars, starts, ends, strict=True):
        if char.isspace():
            if word:
                timings.append(_word(word, word_start, word_end))
                word = ""
            continue
        if not word:
            word_start = float(start)
        word += char
        word_end = float(end)

    if word:
        timings.append(_word(word, word_start, word_end))
    return timings


def _word(word: str, start: float, end: float) -> WordTiming:
    return WordTiming(word=word, start=start, duration=max(0.0, end - start))
