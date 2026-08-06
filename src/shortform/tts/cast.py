"""Per-speaker voice resolution.

Single-narrator strategies resolve one voice for a whole video. Dialogue
strategies — adapted plays, multi-character sketches — need a voice per speaker,
and because cloning a reference voice doesn't scale past a handful of
characters, they need to MIX BACKENDS within one video: F5-TTS clones for the
principals, Edge TTS for bit parts and crowds. A five-hander is a voice per
character; a play with twenty speaking parts is six clones and a lot of Edge.

VoiceCast owns that mapping. Built once per run from the strategy's `tts` block,
it answers two questions: which backend + config does this speaker use, and
give me a cached instance of that backend.

Strategy YAML shape:

    tts:
      backend: "f5_tts"          # the default voice (also the narrator)
      ref_audio: "data/voices/narrator.wav"
      ref_text: "..."
      turn_gap: 0.28
      stage_direction_gap: 0.55
      voices:
        pere_ubu:  { ref_audio: "data/voices/ubu.wav", ref_text: "...", speed: 1.0 }
        bordure:   { backend: "edge", voice: "en-GB-RyanNeural", rate: "+5%" }

Each speaker entry inherits the base config and overrides selectively, so a
clone only has to name its own reference audio. `backend` inside an entry
switches that speaker to a different backend entirely.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shortform.tts.backend import TTSBackend

logger = logging.getLogger(__name__)

DEFAULT_TURN_GAP = 0.28
DEFAULT_STAGE_DIRECTION_GAP = 0.55

# Keys in strategy.tts that configure the cast itself rather than a backend.
# Stripped before the remainder is handed to a backend as its config.
CAST_CONTROL_KEYS = frozenset(
    {"backend", "voices", "turn_gap", "stage_direction_gap"}
)


class UnknownSpeakerError(KeyError):
    """A segment names a speaker the strategy doesn't cast.

    Deliberately fatal rather than falling back to the default voice. A typo'd
    speaker key in a batch run would otherwise narrate an entire character in
    the wrong voice, and you'd only find out by watching the finished video —
    after paying for every second of it.
    """

    def __init__(self, speaker: str, known: list[str]) -> None:
        self.speaker = speaker
        super().__init__(
            f"speaker {speaker!r} is not cast in strategy.tts.voices "
            f"(cast: {', '.join(known) or 'empty'})"
        )


def normalize_speaker(speaker: str) -> str:
    """Canonical speaker key.

    Play texts write 'PÈRE UBU', 'Père Ubu', and 'pere ubu' interchangeably;
    YAML keys want `pere_ubu`. Normalizing both sides means an adaptation can
    emit speaker labels straight out of the source text without a mapping table.
    """
    return "_".join(speaker.strip().lower().replace("-", " ").split())


@dataclass(frozen=True)
class VoiceAssignment:
    """The resolved backend + config for one speaker."""

    speaker: str
    backend_name: str
    config: dict[str, Any]


class VoiceCast:
    """Maps speakers to TTS backends and configs."""

    def __init__(
        self,
        base_config: dict[str, Any],
        default_backend: str,
        voices: dict[str, dict[str, Any]] | None = None,
        backend_factory: Callable[[str], TTSBackend] | None = None,
        turn_gap: float = DEFAULT_TURN_GAP,
        stage_direction_gap: float = DEFAULT_STAGE_DIRECTION_GAP,
    ) -> None:
        self._base_config = {
            k: v for k, v in base_config.items() if k not in CAST_CONTROL_KEYS
        }
        self._default_backend = default_backend
        self._voices = {
            normalize_speaker(k): dict(v or {}) for k, v in (voices or {}).items()
        }
        self._backend_factory = backend_factory or _default_backend_factory
        self.turn_gap = turn_gap
        self.stage_direction_gap = stage_direction_gap

        self._assignments: dict[str, VoiceAssignment] = {}
        self._backends: dict[str, TTSBackend] = {}
        self._default_assignment = VoiceAssignment(
            speaker="",
            backend_name=default_backend,
            config=dict(self._base_config),
        )

    @property
    def has_cast(self) -> bool:
        """Whether the strategy declares any per-speaker voices at all."""
        return bool(self._voices)

    @property
    def speakers(self) -> list[str]:
        return sorted(self._voices)

    def knows(self, speaker: str) -> bool:
        """Whether `speaker` will resolve without raising."""
        if not self._voices:
            return True  # no cast declared → everything uses the single voice
        key = normalize_speaker(speaker)
        return not key or key in self._voices

    def resolve(self, speaker: str) -> VoiceAssignment:
        """The backend + config for `speaker`.

        An empty speaker means the default/narrator voice. Raises
        UnknownSpeakerError for an uncast speaker when a cast IS declared.
        """
        key = normalize_speaker(speaker)
        if not key:
            return self._default_assignment

        cached = self._assignments.get(key)
        if cached is not None:
            return cached

        entry = self._voices.get(key)
        if entry is None:
            if not self._voices:
                # No cast declared — a script that happens to carry turns still
                # works, narrated end-to-end in the single configured voice.
                return self._default_assignment
            raise UnknownSpeakerError(key, self.speakers)

        # The base config carries the default backend's params, so switching a
        # speaker to another backend leaves some inherited keys irrelevant
        # (an `ref_audio` in an Edge config, say). Backends read only the keys
        # they know, so this is harmless — and it keeps entries terse.
        config = {**self._base_config, **entry}
        config.pop("backend", None)
        assignment = VoiceAssignment(
            speaker=key,
            backend_name=str(entry.get("backend") or self._default_backend),
            config=config,
        )
        self._assignments[key] = assignment
        return assignment

    def backend_for(self, backend_name: str) -> TTSBackend:
        """A cached backend instance by name."""
        backend = self._backends.get(backend_name)
        if backend is None:
            backend = self._backend_factory(backend_name)
            self._backends[backend_name] = backend
        return backend

    def gap_after(self, turn_index: int, turns: list[Any]) -> float:
        """Silence to insert after `turns[turn_index]`.

        The gap is longer when the NEXT line carries a stage direction: the beat
        belongs *before* the annotated delivery, not after it. The final turn
        gets no trailing silence — assembly handles spacing between segments.
        """
        if turn_index >= len(turns) - 1:
            return 0.0
        nxt = turns[turn_index + 1]
        if getattr(nxt, "stage_direction", ""):
            return self.stage_direction_gap
        return self.turn_gap

    def summary(self) -> str:
        """One-line description for run logs."""
        if not self._voices:
            return f"single voice ({self._default_backend})"
        parts = []
        for key in self.speakers:
            assignment = self.resolve(key)
            parts.append(f"{key}→{assignment.backend_name}")
        return f"{len(parts)} cast [{', '.join(parts)}], default={self._default_backend}"


def _default_backend_factory(name: str) -> TTSBackend:
    # Imported lazily so this module stays importable without touching the
    # registry (and so tests can inject a factory with no real backends).
    from shortform.tts.registry import get_backend

    return get_backend(name)
