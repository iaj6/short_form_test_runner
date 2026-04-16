"""Pluggable visual generation backends."""

from shortform.visuals.backend import VisualBackend, VisualOutput
from shortform.visuals.registry import get_backend, list_backends

__all__ = ["VisualBackend", "VisualOutput", "get_backend", "list_backends"]
