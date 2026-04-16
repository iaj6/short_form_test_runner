"""Backend registry — maps names to backend instances."""

from __future__ import annotations

from shortform.visuals.backend import VisualBackend
from shortform.visuals.pillow_backend import PillowBackend
from shortform.visuals.veo_backend import VeoBackend

_BACKENDS: dict[str, type] = {
    "pillow": PillowBackend,
    "veo": VeoBackend,
}


def get_backend(name: str, **kwargs: str) -> VisualBackend:
    """Get a visual backend instance by name."""
    cls = _BACKENDS.get(name)
    if cls is None:
        available = ", ".join(_BACKENDS)
        raise ValueError(f"Unknown visual backend '{name}'. Available: {available}")
    return cls(**kwargs)  # type: ignore[return-value]


def list_backends() -> list[str]:
    """List available backend names."""
    return list(_BACKENDS)
