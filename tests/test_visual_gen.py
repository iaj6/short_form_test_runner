"""Tests for visual_gen helpers — per-segment camera move injection (#3)."""

from __future__ import annotations

from shortform.stages.visual_gen import _apply_camera_move


def test_camera_move_noop_without_moves():
    cfg = {"animation_style": "stop-motion claymation"}
    _apply_camera_move(cfg, 0)
    assert cfg["animation_style"] == "stop-motion claymation"


def test_camera_move_prepends_and_cycles():
    base = "stop-motion claymation, candlelight"
    moves = ["push-in", "dolly-out", "locked shot"]
    styles = []
    for i in range(4):
        cfg = {"animation_style": base, "camera_moves": moves}
        _apply_camera_move(cfg, i)
        styles.append(cfg["animation_style"])

    # Each segment gets a distinct move prepended to the base look.
    assert styles[0] == "push-in, " + base
    assert styles[1] == "dolly-out, " + base
    assert styles[2] == "locked shot, " + base
    # Cycles back around (index 3 % 3 == 0).
    assert styles[3] == "push-in, " + base


def test_camera_move_without_base_style():
    cfg = {"camera_moves": ["slow pan"]}
    _apply_camera_move(cfg, 0)
    assert cfg["animation_style"] == "slow pan"
