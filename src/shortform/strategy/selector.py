"""Strategy selector — multi-armed bandit (Phase 4)."""

from __future__ import annotations

import random

from shortform.config import list_strategies


def select_strategy(exploit_ratio: float = 0.7) -> str:
    """For now, just pick a random strategy. Phase 4 adds bandit logic."""
    strategies = list_strategies()
    if not strategies:
        raise ValueError("No strategies found in config/strategies/")
    return random.choice(strategies)
