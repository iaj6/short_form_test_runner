"""Tests for music track selection.

The two claims worth defending: a track that fits the content wins, and the
same episode gets the same track on every re-render (assembly resumes and
re-renders, so `random.choice` scored a resumed batch inconsistently).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from shortform.music.selector import (
    MusicRequest,
    Track,
    load_tracks,
    score_track,
    select_track,
)


def _library(tmp_path: Path, entries: list[dict], make_files: bool = True) -> Path:
    """A category directory with a manifest and (optionally) its audio."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for e in entries:
        if make_files:
            (tmp_path / e["file"]).write_bytes(b"audio")
    (tmp_path / "tracks.yaml").write_text(yaml.safe_dump({"tracks": entries}))
    return tmp_path


PIANO = {
    "file": "piano.mp3", "title": "Victorian Piano", "mood": ["victorian", "piano", "melancholic"],
    "tempo": "slow", "intensity": "low", "attribution_required": False,
}
MUSIC_BOX = {
    "file": "box.mp3", "title": "Dark Music Box", "mood": ["music_box", "spooky", "gothic"],
    "tempo": "slow", "intensity": "medium", "attribution_required": False,
}
CLOCKWORK = {
    "file": "clock.mp3", "title": "Clockwork", "mood": ["mechanical", "clockwork"],
    "tempo": "medium", "intensity": "medium", "attribution_required": False,
}


# --- Manifest loading -------------------------------------------------------


def test_loads_tracks_with_metadata(tmp_path: Path):
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX])
    tracks = load_tracks(d)
    assert {t.path.name for t in tracks} == {"piano.mp3", "box.mp3"}
    assert tracks[0].mood == ["victorian", "piano", "melancholic"]


def test_entry_without_audio_is_skipped(tmp_path: Path):
    """Manifests are committed but audio is gitignored — a fresh clone has
    entries with no file behind them, and that must not crash a render."""
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX], make_files=False)
    (d / "piano.mp3").write_bytes(b"audio")  # only one of them present
    tracks = load_tracks(d)
    assert [t.path.name for t in tracks] == ["piano.mp3"]


def test_no_manifest_returns_empty(tmp_path: Path):
    d = tmp_path / "ambient"
    d.mkdir()
    assert load_tracks(d) == []


def test_corrupt_manifest_degrades_to_glob(tmp_path: Path):
    d = tmp_path / "gothic"
    d.mkdir()
    (d / "a.mp3").write_bytes(b"audio")
    (d / "tracks.yaml").write_text("{ not: valid: yaml:")
    assert load_tracks(d) == []
    # …and selection still returns something rather than failing the render.
    assert select_track(d, MusicRequest(seed="x")) is not None


# --- Scoring ----------------------------------------------------------------


def test_mood_overlap_scores():
    t = Track(path=Path("x.mp3"), mood=["gothic", "piano"], tempo="slow", intensity="low")
    assert score_track(t, MusicRequest(mood=["gothic"])) > 0
    assert score_track(t, MusicRequest(mood=["gothic", "piano"])) > score_track(
        t, MusicRequest(mood=["gothic"])
    )
    assert score_track(t, MusicRequest(mood=["upbeat"])) == 0


def test_mood_outweighs_tempo():
    """A cue that feels right beats one that merely runs at the right speed."""
    right_mood = Track(path=Path("a.mp3"), mood=["gothic"], tempo="fast")
    right_tempo = Track(path=Path("b.mp3"), mood=["upbeat"], tempo="slow")
    req = MusicRequest(mood=["gothic"], tempo="slow")
    assert score_track(right_mood, req) > score_track(right_tempo, req)


def test_matching_is_case_insensitive():
    t = Track(path=Path("x.mp3"), mood=["Gothic"], tempo="Slow")
    assert score_track(t, MusicRequest(mood=["gothic"], tempo="slow")) > 0


# --- Selection --------------------------------------------------------------


def test_picks_the_best_match(tmp_path: Path):
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX, CLOCKWORK])
    chosen = select_track(d, MusicRequest(mood=["music_box", "spooky"], seed="ep01"))
    assert chosen is not None and chosen.path.name == "box.mp3"


def test_same_episode_always_gets_the_same_track(tmp_path: Path):
    """The re-render stability property. With random.choice a resumed batch
    re-scored the same episode differently on every attempt."""
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX, CLOCKWORK])
    picks = {
        select_track(d, MusicRequest(seed="uburex01e01")).path.name  # type: ignore[union-attr]
        for _ in range(8)
    }
    assert len(picks) == 1


def test_different_episodes_spread_across_the_library(tmp_path: Path):
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX, CLOCKWORK])
    picks = {
        select_track(d, MusicRequest(seed=f"ep{i:02d}")).path.name  # type: ignore[union-attr]
        for i in range(24)
    }
    assert len(picks) > 1, "seeding must vary across episodes, not pin one track"


def test_unmatched_mood_still_returns_a_track(tmp_path: Path):
    """Scoring is a preference, not a filter — a strategy asking for moods no
    track carries should get the closest thing, not silence."""
    d = _library(tmp_path / "gothic", [PIANO, MUSIC_BOX])
    assert select_track(d, MusicRequest(mood=["polka"], seed="x")) is not None


def test_empty_category_returns_none(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    assert select_track(d, MusicRequest(seed="x")) is None


def test_uncurated_category_falls_back_to_glob(tmp_path: Path):
    """ambient/ and upbeat/ have no manifest and must keep working."""
    d = tmp_path / "ambient"
    d.mkdir()
    (d / "a.mp3").write_bytes(b"audio")
    (d / "b.mp3").write_bytes(b"audio")
    chosen = select_track(d, MusicRequest(seed="ep01"))
    assert chosen is not None and chosen.path.suffix == ".mp3"
    # Still stable across re-renders.
    assert select_track(d, MusicRequest(seed="ep01")).path == chosen.path  # type: ignore[union-attr]


# --- Attribution ------------------------------------------------------------


def test_no_credit_when_not_required():
    assert Track(path=Path("x.mp3"), attribution_required=False).credit == ""


def test_credit_uses_supplied_text():
    t = Track(
        path=Path("x.mp3"), attribution_required=True,
        attribution_text="Track by Someone (CC BY 4.0)",
    )
    assert t.credit == "Track by Someone (CC BY 4.0)"


def test_credit_falls_back_when_text_missing():
    """A track flagged as needing credit but carrying none is a curation gap —
    emit something usable rather than silently crediting no one."""
    t = Track(
        path=Path("x.mp3"), attribution_required=True,
        title="Some Title", composer="Some Artist",
    )
    assert "Some Title" in t.credit and "Some Artist" in t.credit


# --- Episode mood overrides the strategy default ----------------------------


def test_episode_mood_wins_over_strategy_default(tmp_path: Path, monkeypatch):
    """The strategy default is the series signature; the per-episode value is
    the scene's colour. A serial needs both — a strategy-only mood would pin
    every episode to one cue and leave the rest of the library unused."""
    from shortform.config import MusicConfig
    from shortform.stages import assembly

    music_root = tmp_path / "music"
    _library(music_root / "ubu", [PIANO, MUSIC_BOX, CLOCKWORK])
    monkeypatch.setattr(assembly, "PROJECT_ROOT", tmp_path)

    cfg = MusicConfig(music_dir="music")
    strategy_music = {"mood": ["victorian", "piano"]}

    # No episode mood -> the strategy default (the "signature" cue).
    default_pick = assembly._pick_music_track(
        cfg, "ubu", strategy_music=strategy_music, seed="ep01"
    )
    assert default_pick is not None and default_pick.path.name == "piano.mp3"

    # Episode mood overrides it entirely.
    scene_pick = assembly._pick_music_track(
        cfg, "ubu", strategy_music=strategy_music, seed="ep01",
        episode_mood=["music_box", "spooky"],
    )
    assert scene_pick is not None and scene_pick.path.name == "box.mp3"


def test_missing_category_returns_none(tmp_path: Path, monkeypatch):
    from shortform.config import MusicConfig
    from shortform.stages import assembly

    monkeypatch.setattr(assembly, "PROJECT_ROOT", tmp_path)
    assert assembly._pick_music_track(MusicConfig(music_dir="music"), "nope") is None
