"""Tests for the config system."""

from shortform.config import list_strategies, load_settings, load_strategy


def test_load_settings():
    settings = load_settings()
    assert settings.video.width == 1080
    assert settings.video.height == 1920
    assert settings.video.fps == 30
    assert settings.llm.model == "claude-sonnet-4-20250514"


def test_list_strategies():
    strategies = list_strategies()
    assert "motivation_quotes" in strategies
    assert "tech_tips" in strategies


def test_load_strategy():
    strat = load_strategy("motivation_quotes")
    assert strat.name == "motivation_quotes"
    assert strat.category == "motivation"
    assert len(strat.topics) > 0
    assert "system" in strat.prompts
    assert "template" in strat.prompts
