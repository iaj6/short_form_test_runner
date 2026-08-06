"""Tests for the unattended batch runner.

The behaviours that make an overnight run trustworthy: a failure isolates to its
episode, depleted credits abort instead of burning wall-clock, finished episodes
are skipped on a re-run, and the report surfaces what needs a human.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shortform.pipeline.batch import (
    BatchReport,
    EpisodeResult,
    EpisodeStatus,
    critic_findings,
    is_credits_error,
    run_batch,
)

CREDITS_ERROR = (
    "Stage visual_gen failed: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
    "'message': 'Your prepayment credits are depleted.'}}"
)


def _scripts(tmp_path: Path, n: int) -> list[Path]:
    out = []
    for i in range(1, n + 1):
        p = tmp_path / f"ep{i:02d}.json"
        p.write_text("{}")
        out.append(p)
    return out


def _ok(path: Path, duration: float = 70.0, **kw) -> EpisodeResult:
    return EpisodeResult(
        script_path=path, script_id=path.stem, status=EpisodeStatus.COMPLETED,
        duration=duration, output_path=f"/videos/{path.stem}.mp4", **kw
    )


def _fail(path: Path, error: str) -> EpisodeResult:
    return EpisodeResult(
        script_path=path, script_id=path.stem,
        status=EpisodeStatus.FAILED, error=error,
    )


# --- Credits detection ------------------------------------------------------


def test_credits_error_recognised():
    """Must match veo_backend's sniff — both need the same answer."""
    assert is_credits_error(CREDITS_ERROR)
    assert is_credits_error("Your prepayment credits are depleted")
    assert is_credits_error("please check your BILLING settings")


def test_ordinary_errors_are_not_credits():
    assert not is_credits_error("429 RESOURCE_EXHAUSTED: quota exceeded, retry later")
    assert not is_credits_error("ffmpeg failed: no such file")
    assert not is_credits_error("")


# --- Orchestration ----------------------------------------------------------


@pytest.mark.asyncio
async def test_renders_every_script(tmp_path: Path):
    scripts = _scripts(tmp_path, 3)
    report = await run_batch(scripts, render=lambda p: _async(_ok(p)))

    assert len(report.completed) == 3
    assert report.ok
    assert [r.script_id for r in report.results] == ["ep01", "ep02", "ep03"]


@pytest.mark.asyncio
async def test_one_failure_does_not_stop_the_batch(tmp_path: Path):
    """The whole point of a batch: episode 2 dying must not cost you 3."""
    scripts = _scripts(tmp_path, 3)

    async def render(p: Path) -> EpisodeResult:
        return _fail(p, "ffmpeg exploded") if p.stem == "ep02" else _ok(p)

    report = await run_batch(scripts, render=render)

    assert len(report.completed) == 2
    assert [r.script_id for r in report.failed] == ["ep02"]
    assert not report.not_attempted, "later episodes must still be attempted"
    assert not report.ok


@pytest.mark.asyncio
async def test_render_exception_is_caught_as_a_failure(tmp_path: Path):
    """An unexpected raise must be contained, not propagate out of the batch."""
    scripts = _scripts(tmp_path, 2)

    async def render(p: Path) -> EpisodeResult:
        if p.stem == "ep01":
            raise RuntimeError("unhandled boom")
        return _ok(p)

    report = await run_batch(scripts, render=render)

    assert len(report.failed) == 1
    assert "unhandled boom" in report.failed[0].error
    assert len(report.completed) == 1


@pytest.mark.asyncio
async def test_credits_depleted_aborts_the_rest(tmp_path: Path):
    """Every remaining episode would fail identically — don't spend hours proving it."""
    scripts = _scripts(tmp_path, 5)
    attempted: list[str] = []

    async def render(p: Path) -> EpisodeResult:
        attempted.append(p.stem)
        return _fail(p, CREDITS_ERROR) if p.stem == "ep02" else _ok(p)

    report = await run_batch(scripts, render=render)

    assert attempted == ["ep01", "ep02"], "must stop dead, not try ep03-05"
    assert [r.script_id for r in report.not_attempted] == ["ep03", "ep04", "ep05"]
    assert "credits" in report.aborted_reason.lower()
    assert not report.ok


@pytest.mark.asyncio
async def test_ordinary_failure_does_not_abort(tmp_path: Path):
    scripts = _scripts(tmp_path, 3)
    attempted: list[str] = []

    async def render(p: Path) -> EpisodeResult:
        attempted.append(p.stem)
        return _fail(p, "some transient thing") if p.stem == "ep01" else _ok(p)

    report = await run_batch(scripts, render=render)
    assert attempted == ["ep01", "ep02", "ep03"]
    assert not report.aborted_reason


@pytest.mark.asyncio
async def test_stop_on_failure_halts(tmp_path: Path):
    scripts = _scripts(tmp_path, 3)

    async def render(p: Path) -> EpisodeResult:
        return _fail(p, "nope") if p.stem == "ep01" else _ok(p)

    report = await run_batch(scripts, render=render, stop_on_failure=True)
    assert len(report.not_attempted) == 2
    assert "stop-on-failure" in report.aborted_reason


@pytest.mark.asyncio
async def test_finished_episodes_are_skipped(tmp_path: Path):
    """Re-running a crashed batch must cost nothing for what already worked."""
    scripts = _scripts(tmp_path, 3)
    rendered: list[str] = []

    async def render(p: Path) -> EpisodeResult:
        rendered.append(p.stem)
        return _ok(p)

    def already(p: Path) -> str:
        return f"/videos/{p.stem}.mp4" if p.stem in {"ep01", "ep03"} else ""

    report = await run_batch(scripts, render=render, already_rendered=already)

    assert rendered == ["ep02"], "only the unfinished episode should render"
    assert len(report.skipped) == 2
    assert report.ok


# --- Critic findings --------------------------------------------------------


def test_only_the_final_attempt_counts_as_flagged():
    """A rejection that a retry fixed is the system working, not a problem."""
    flagged, unverified = critic_findings({
        "critic_reviews": [
            {"clip": "a.mp4", "attempt": 1, "passed": False, "unverified": False},
            {"clip": "a.mp4", "attempt": 2, "passed": True, "unverified": False},
        ]
    })
    assert flagged == []
    assert unverified == []


def test_clip_failing_every_attempt_is_flagged():
    flagged, _ = critic_findings({
        "critic_reviews": [
            {"clip": "b.mp4", "attempt": 1, "passed": False, "unverified": False},
            {"clip": "b.mp4", "attempt": 2, "passed": False, "unverified": False},
        ]
    })
    assert flagged == ["b.mp4"]


def test_unverified_tracked_separately_from_flagged():
    """'never checked' and 'checked and failed' are different problems."""
    flagged, unverified = critic_findings({
        "critic_reviews": [
            {"clip": "c.mp4", "attempt": 1, "passed": True, "unverified": True},
        ]
    })
    assert flagged == []
    assert unverified == ["c.mp4"]


def test_no_reviews_is_clean():
    assert critic_findings({}) == ([], [])


# --- Report -----------------------------------------------------------------


def test_report_surfaces_review_items(tmp_path: Path):
    scripts = _scripts(tmp_path, 2)
    report = BatchReport(results=[
        _ok(scripts[0], flagged=["segment_01_clip_02.mp4"]),
        _ok(scripts[1], unverified=["segment_00.mp4"]),
    ])
    text = "\n".join(report.summary_lines())

    assert "NEEDS HUMAN REVIEW" in text
    assert "segment_01_clip_02.mp4" in text
    assert "1 clip(s) never verified" in text


def test_report_ok_despite_review_items(tmp_path: Path):
    """A flagged clip still rendered — the batch succeeded, a human just looks."""
    scripts = _scripts(tmp_path, 1)
    report = BatchReport(results=[_ok(scripts[0], flagged=["x.mp4"])])
    assert report.ok
    assert report.needs_review


def test_report_serializes(tmp_path: Path):
    scripts = _scripts(tmp_path, 1)
    data = BatchReport(results=[_ok(scripts[0])]).to_dict()
    assert data["completed"] == 1
    assert data["episodes"][0]["status"] == "completed"


async def _async(value: EpisodeResult) -> EpisodeResult:
    return value


# --- Backend-aware skip -----------------------------------------------------


def _rendered(videos_dir: Path, script_id: str, backend: str) -> Path:
    """Simulate a finished episode: a video plus the manifest recording it."""
    from shortform.pipeline.batch import write_manifest

    videos_dir.mkdir(parents=True, exist_ok=True)
    video = videos_dir / f"{script_id}_Some Title.mp4"
    video.write_bytes(b"video")
    write_manifest(
        videos_dir,
        EpisodeResult(
            script_path=Path(f"{script_id}.json"), script_id=script_id,
            status=EpisodeStatus.COMPLETED, output_path=str(video),
        ),
        backend=backend,
    )
    return video


def test_matching_backend_is_reused(tmp_path: Path):
    from shortform.pipeline.batch import reusable_output

    video = _rendered(tmp_path, "ep01", "veo")
    assert reusable_output(tmp_path, "ep01", "veo") == str(video)


def test_different_backend_is_rerendered(tmp_path: Path):
    """THE bug this fixes: a cheap pillow test must not block the real Veo pass."""
    from shortform.pipeline.batch import reusable_output

    _rendered(tmp_path, "ep01", "pillow")
    assert reusable_output(tmp_path, "ep01", "veo") == ""


def test_video_without_manifest_is_rerendered(tmp_path: Path):
    """Backend unknown, so we can't claim it's done — err toward rendering.
    A wasted re-render costs money; a wrong skip produces nothing at all."""
    from shortform.pipeline.batch import reusable_output

    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ep01_Some Title.mp4").write_bytes(b"video")
    assert reusable_output(tmp_path, "ep01", "veo") == ""


def test_manifest_without_video_is_rerendered(tmp_path: Path):
    """Someone deleted the mp4 to force a redo — honour that."""
    from shortform.pipeline.batch import reusable_output

    video = _rendered(tmp_path, "ep01", "veo")
    video.unlink()
    assert reusable_output(tmp_path, "ep01", "veo") == ""


def test_empty_video_is_rerendered(tmp_path: Path):
    from shortform.pipeline.batch import reusable_output

    video = _rendered(tmp_path, "ep01", "veo")
    video.write_bytes(b"")
    assert reusable_output(tmp_path, "ep01", "veo") == ""


def test_corrupt_manifest_is_rerendered(tmp_path: Path):
    from shortform.pipeline.batch import manifest_path, reusable_output

    _rendered(tmp_path, "ep01", "veo")
    manifest_path(tmp_path, "ep01").write_text("{ not json")
    assert reusable_output(tmp_path, "ep01", "veo") == ""


def test_manifest_records_critic_findings(tmp_path: Path):
    from shortform.pipeline.batch import read_manifest, write_manifest

    write_manifest(
        tmp_path,
        EpisodeResult(
            script_path=Path("ep01.json"), script_id="ep01",
            status=EpisodeStatus.COMPLETED, output_path=str(tmp_path / "ep01_T.mp4"),
            flagged=["segment_01_clip_02.mp4"], unverified=["segment_00.mp4"],
        ),
        backend="veo",
    )
    m = read_manifest(tmp_path, "ep01")
    assert m is not None
    assert m["backend"] == "veo"
    assert m["flagged"] == ["segment_01_clip_02.mp4"]
    assert m["unverified"] == ["segment_00.mp4"]
    assert m["rendered_at"]
