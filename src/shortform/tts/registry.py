"""TTS backend registry — maps names to backend classes."""

from __future__ import annotations

from typing import Any

from shortform.tts.backend import TTSBackend
from shortform.tts.edge_backend import EdgeBackend
from shortform.tts.f5_backend import F5TTSBackend

_BACKENDS: dict[str, type] = {
    "edge": EdgeBackend,
    "f5_tts": F5TTSBackend,
}


def get_backend(name: str, **kwargs: Any) -> TTSBackend:
    """Get a TTS backend instance by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        available = ", ".join(_BACKENDS)
        raise ValueError(f"Unknown TTS backend '{name}'. Available: {available}")
    return cls(**kwargs)  # type: ignore[return-value]


def list_backends() -> list[str]:
    return list(_BACKENDS)
