"""Batch runner — render many episodes unattended.

Rendering one episode is a foreground operation a human watches. Rendering
ninety is not: it runs overnight, something will go wrong partway through, and
the person who started it needs to know in the morning which episodes are
finished, which need another pass, and which need a human to look at them.

The pieces that make that safe:

**Deterministic video ids.** `generate-from-script` mints a random id per run, so
its assets land in a fresh directory and a re-run regenerates everything. In
batch mode the video id IS the script id, so assets are addressable
(`data/assets/uburex01e01/`), outputs sort by episode, and a re-run automatically
reuses every clip the previous attempt paid for. Without this, resuming a
half-finished batch costs full price a second time.

**Skip what's done.** An episode with a finished MP4 on disk is skipped outright,
so re-running a batch after a crash costs nothing for the parts that succeeded.

**Failure isolation, with one exception.** A single episode failing must not kill
the batch — the rest are independent. Depleted credits are the exception: every
subsequent episode would fail identically, so the batch aborts and reports what
it didn't attempt rather than burning wall-clock on ninety guaranteed failures.

**A report that leads with what needs a human.** Critic flags and unverified
clips are the whole reason the batch can run unattended; burying them under a
success count would defeat the point.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EpisodeStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"      # already rendered
    NOT_ATTEMPTED = "not_attempted"  # batch aborted before reaching it


@dataclass
class EpisodeResult:
    """Outcome of one episode in the batch."""

    script_path: Path
    script_id: str
    title: str = ""
    status: EpisodeStatus = EpisodeStatus.NOT_ATTEMPTED
    duration: float = 0.0
    output_path: str = ""
    error: str = ""
    # Clips the critic flagged after exhausting its retries, and clips it
    # couldn't check at all. Both need a human; they're tracked separately
    # because "flagged" and "never verified" are different problems.
    flagged: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.flagged or self.unverified)

    def to_dict(self) -> dict[str, Any]:
        return {
            "script": str(self.script_path),
            "script_id": self.script_id,
            "title": self.title,
            "status": self.status.value,
            "duration": round(self.duration, 1),
            "output": self.output_path,
            "error": self.error,
            "flagged": self.flagged,
            "unverified": self.unverified,
        }


@dataclass
class BatchReport:
    results: list[EpisodeResult] = field(default_factory=list)
    aborted_reason: str = ""

    def _of(self, status: EpisodeStatus) -> list[EpisodeResult]:
        return [r for r in self.results if r.status == status]

    @property
    def completed(self) -> list[EpisodeResult]:
        return self._of(EpisodeStatus.COMPLETED)

    @property
    def failed(self) -> list[EpisodeResult]:
        return self._of(EpisodeStatus.FAILED)

    @property
    def skipped(self) -> list[EpisodeResult]:
        return self._of(EpisodeStatus.SKIPPED)

    @property
    def not_attempted(self) -> list[EpisodeResult]:
        return self._of(EpisodeStatus.NOT_ATTEMPTED)

    @property
    def needs_review(self) -> list[EpisodeResult]:
        return [r for r in self.results if r.needs_review]

    @property
    def ok(self) -> bool:
        """Whether the batch finished cleanly. Episodes needing review still
        count as ok — they rendered; a human just has to look at them."""
        return not self.failed and not self.not_attempted and not self.aborted_reason

    def summary_lines(self) -> list[str]:
        """Human-readable report, ordered so the actionable parts come last —
        that's what's on screen when a long batch finishes."""
        lines: list[str] = []
        total_runtime = sum(r.duration for r in self.completed)
        lines.append(
            f"Batch: {len(self.completed)} rendered, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed, {len(self.not_attempted)} not attempted "
            f"({total_runtime / 60:.1f} min of video)"
        )

        for r in self.results:
            mark = {
                EpisodeStatus.COMPLETED: "ok  ",
                EpisodeStatus.SKIPPED: "skip",
                EpisodeStatus.FAILED: "FAIL",
                EpisodeStatus.NOT_ATTEMPTED: "----",
            }[r.status]
            detail = ""
            if r.status == EpisodeStatus.COMPLETED:
                detail = f"{r.duration:5.1f}s  {Path(r.output_path).name}"
            elif r.status == EpisodeStatus.FAILED:
                detail = r.error[:100]
            lines.append(f"  [{mark}] {r.script_id:<16} {detail}")

        if self.aborted_reason:
            lines.append("")
            lines.append(f"BATCH ABORTED: {self.aborted_reason}")

        if self.needs_review:
            lines.append("")
            lines.append("NEEDS HUMAN REVIEW:")
            for r in self.needs_review:
                for clip in r.flagged:
                    lines.append(f"  {r.script_id}: {clip} — failed continuity")
                if r.unverified:
                    lines.append(
                        f"  {r.script_id}: {len(r.unverified)} clip(s) never verified"
                    )
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": len(self.completed),
            "skipped": len(self.skipped),
            "failed": len(self.failed),
            "not_attempted": len(self.not_attempted),
            "aborted_reason": self.aborted_reason,
            "episodes": [r.to_dict() for r in self.results],
        }


MANIFEST_SUFFIX = ".manifest.json"


def manifest_path(videos_dir: Path, script_id: str) -> Path:
    return videos_dir / f"{script_id}{MANIFEST_SUFFIX}"


def write_manifest(videos_dir: Path, result: EpisodeResult, backend: str) -> None:
    """Record what produced this episode, beside the video it describes.

    Exists so the skip check can ask "was this made with the backend I'm asking
    for?" rather than just "does a file exist". Without it a cheap `-vb pillow`
    test run silently blocks a later Veo render of the same episodes — the
    output is present, so the episode looks finished.
    """
    videos_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "script_id": result.script_id,
        "backend": backend,
        "output": Path(result.output_path).name,
        "title": result.title,
        "duration": round(result.duration, 1),
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
        "flagged": result.flagged,
        "unverified": result.unverified,
    }
    manifest_path(videos_dir, result.script_id).write_text(
        json.dumps(payload, indent=2)
    )


def read_manifest(videos_dir: Path, script_id: str) -> dict[str, Any] | None:
    path = manifest_path(videos_dir, script_id)
    if not path.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except (OSError, ValueError):
        logger.warning("Unreadable manifest %s — treating as unrendered", path.name)
        return None


def reusable_output(videos_dir: Path, script_id: str, backend: str) -> str:
    """An existing video for this episode that was made with `backend`, else "".

    Four cases, and the default in every ambiguous one is to RENDER — a wasted
    re-render costs money, but a wrong skip produces nothing at all and isn't
    noticed until someone looks for the file.

    - No manifest: can't know what produced it. Render.
      (Also covers outputs from `generate-from-script`, which has its own id
      scheme and writes no manifest.)
    - Manifest but the video is gone: render.
    - Manifest from a different backend: render, and say so — a `-vb pillow`
      test must not block the real Veo pass.
    - Manifest, video present, backend matches: skip.
    """
    manifest = read_manifest(videos_dir, script_id)
    if manifest is None:
        # Don't skip on a bare file match — we can't tell what produced it.
        for candidate in sorted(videos_dir.glob(f"{script_id}_*.mp4")):
            if candidate.stat().st_size > 0:
                logger.info(
                    "%s: existing video has no manifest (backend unknown) — "
                    "re-rendering", script_id,
                )
                break
        return ""

    output = videos_dir / str(manifest.get("output", ""))
    if not output.exists() or output.stat().st_size == 0:
        logger.info("%s: manifest present but video missing — re-rendering", script_id)
        return ""

    previous = str(manifest.get("backend", ""))
    if previous != backend:
        logger.warning(
            "%s: existing video was made with '%s', now rendering with '%s' — "
            "the previous output will be replaced",
            script_id, previous or "unknown", backend,
        )
        return ""

    return str(output)


def is_credits_error(message: str) -> bool:
    """Whether a failure means the account is out of credits.

    Matches the sniff in veo_backend's retry ladder — both need the same
    answer, and a batch that kept going after credits ran dry would spend its
    remaining wall-clock failing identically on every episode.
    """
    lowered = message.lower()
    return (
        "credits are depleted" in lowered
        or "prepayment credits" in lowered
        or "billing" in lowered
    )


def critic_findings(artifacts: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(flagged, unverified) clip names from a run's critic reviews.

    A clip is only *flagged* if its FINAL attempt failed — an early rejection
    that a retry fixed is the system working, not a problem to report.
    """
    reviews = artifacts.get("critic_reviews") or []
    final: dict[str, dict[str, Any]] = {}
    for review in reviews:
        final[str(review.get("clip", ""))] = review

    flagged = [c for c, r in final.items() if not r.get("passed", True)]
    unverified = [c for c, r in final.items() if r.get("unverified")]
    return sorted(flagged), sorted(unverified)


async def run_batch(
    scripts: list[Path],
    render: Callable[[Path], Awaitable[EpisodeResult]],
    already_rendered: Callable[[Path], str] | None = None,
    stop_on_failure: bool = False,
) -> BatchReport:
    """Render each script in sequence, isolating failures.

    `render` does the actual work for one episode. `already_rendered` returns an
    existing output path for a script (or "" to render it), which is what makes
    re-running a crashed batch nearly free.

    Sequential on purpose: Veo is rate-limited, and a batch you can Ctrl-C
    without leaving half-written parallel state is worth more than the wall-clock
    a concurrent version would save.
    """
    report = BatchReport(
        results=[EpisodeResult(script_path=p, script_id=p.stem) for p in scripts]
    )

    for index, result in enumerate(report.results):
        existing = already_rendered(result.script_path) if already_rendered else ""
        if existing:
            result.status = EpisodeStatus.SKIPPED
            result.output_path = existing
            logger.info(
                "[%d/%d] %s: already rendered — skipping",
                index + 1, len(scripts), result.script_id,
            )
            continue

        logger.info(
            "[%d/%d] %s: rendering", index + 1, len(scripts), result.script_id
        )
        try:
            rendered = await render(result.script_path)
        except Exception as e:  # noqa: BLE001 — one episode must not kill the batch
            rendered = EpisodeResult(
                script_path=result.script_path,
                script_id=result.script_id,
                status=EpisodeStatus.FAILED,
                error=str(e),
            )

        # Preserve identity/order; copy the outcome onto the placeholder.
        result.status = rendered.status
        result.title = rendered.title
        result.duration = rendered.duration
        result.output_path = rendered.output_path
        result.error = rendered.error
        result.flagged = rendered.flagged
        result.unverified = rendered.unverified

        if result.status == EpisodeStatus.FAILED:
            logger.error("%s FAILED: %s", result.script_id, result.error)
            if is_credits_error(result.error):
                report.aborted_reason = (
                    "Veo credits depleted — remaining episodes would fail "
                    "identically. Top up, then re-run; finished episodes are "
                    "skipped and partial ones resume from their existing clips."
                )
                break
            if stop_on_failure:
                report.aborted_reason = (
                    f"{result.script_id} failed and --stop-on-failure was set"
                )
                break
        elif result.needs_review:
            logger.warning(
                "%s rendered but needs review: %d flagged, %d unverified",
                result.script_id, len(result.flagged), len(result.unverified),
            )

    return report
