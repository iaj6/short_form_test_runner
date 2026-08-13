"""Visual generation stage — delegates to pluggable backends.

For video-output backends (Veo), we generate N clips per segment so the
visual track is at least as long as the F5-TTS narration. Veo is hard-locked
at ~8s per clip, so a 20s narration needs 3 clips. Still-image backends
(Pillow) keep producing one asset per segment — Ken Burns in assembly
extends to any duration.
"""

from __future__ import annotations

import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from shortform.config import PROJECT_ROOT
from shortform.models.script import Segment
from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.tts.cast import normalize_speaker
from shortform.visuals.backend import VisualBackend, VisualOutput, VisualOutputType
from shortform.visuals.critic import ClipCritic
from shortform.visuals.pillow_backend import PillowBackend

logger = logging.getLogger(__name__)

# Approximate usable seconds per Veo clip when stitched with a small xfade.
# Veo 3 produces ~8s clips; we leave a bit of headroom for the inter-clip
# crossfade in assembly, so each clip "contributes" ~7.5s of timeline.
CLIP_TARGET_SECONDS = 7.5

# Generation attempts per clip when the critic rejects it. Attempt 3 drops the
# chain anchor and re-anchors to the hero reference — see _generate_reviewed.
CRITIC_MAX_ATTEMPTS = 3

# Records which backend produced the clips in a working directory. Clip reuse
# without this is backend-blind: a Veo run resuming into a directory left by a
# cheap Pillow pass silently reuses the stills, calls Veo zero times, and
# reports success. Worse, the batch runner then writes a manifest claiming the
# episode was rendered with Veo — so the lie persists and every later run skips
# it. Cheap file, expensive absence.
PROVENANCE_FILE = ".visual_backend"

# Clips the critic rejected on every attempt and which were kept anyway. Reuse
# checks readability, not correctness, so without this record a resume adopts a
# clip the critic already condemned — and the episode ships with it, since a
# reused clip is never re-reviewed. One line per clip filename.
FLAGGED_FILE = ".flagged_clips"


class VisualGenStage:
    def __init__(
        self,
        backend: VisualBackend | None = None,
        reuse_existing: bool = True,
        critic: ClipCritic | None = None,
    ) -> None:
        self._backend = backend or PillowBackend()
        # Optional continuity critic. Without one the stage generates blind —
        # fine when a human reviews every render, wrong for unattended batches.
        self._critic = critic
        # Reuse already-generated clips found in the working directory. Veo is by
        # far the most expensive stage and interrupted runs are routine (depleted
        # credits, safety-filter bailouts, rate limits) — without this, a run that
        # dies on the last clip throws away every clip before it. Only has an
        # effect when the run resumes into an existing video id
        # (`generate-from-script --video-id`), since a fresh id gets a fresh dir.
        self._reuse_existing = reuse_existing
        # Set per run in execute(), once the working directory is known.
        self._reuse_this_run = False
        self._run_dir: Path | None = None
        self._flagged: set[str] = set()

    @property
    def name(self) -> str:
        return "visual_gen"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        if not ctx.script.segments:
            errors.append("No script segments for visual generation")
        return errors

    def _reuse_allowed(self, video_dir: Path) -> bool:
        """Whether clips already in `video_dir` were made by this backend.

        Written at the START of a run so an interrupted one still resumes into
        its own clips. A missing marker means the directory predates this check
        and its provenance is unknown — regenerate, because a wasted re-render
        costs money while a wrong reuse silently ships the wrong video.
        """
        if not self._reuse_existing:
            return False

        marker = video_dir / PROVENANCE_FILE
        previous = marker.read_text().strip() if marker.exists() else ""
        if previous == self._backend.name:
            return True

        if previous:
            logger.warning(
                "Existing clips in %s were made with '%s', not '%s' — "
                "regenerating rather than reusing them",
                video_dir.name, previous, self._backend.name,
            )
            return False

        if any(video_dir.glob("segment_*")):
            logger.warning(
                "Existing clips in %s have no recorded backend — regenerating "
                "rather than assuming they match '%s'",
                video_dir.name, self._backend.name,
            )
            return False

        # Fresh directory: nothing to conflict with, and nothing to reuse.
        return True

    def _load_flagged(self, video_dir: Path) -> set[str]:
        marker = video_dir / FLAGGED_FILE
        if not marker.exists():
            return set()
        return {line.strip() for line in marker.read_text().splitlines() if line.strip()}

    def _save_flagged(self) -> None:
        """Persist the flagged set, if this stage is running against a directory.

        A no-op when `_run_dir` is unset, which is how the unit tests drive
        `_generate_reviewed` directly — the set still tracks in memory.
        """
        if self._run_dir is None:
            return
        marker = self._run_dir / FLAGGED_FILE
        if not self._flagged:
            marker.unlink(missing_ok=True)
            return
        marker.write_text("\n".join(sorted(self._flagged)) + "\n")

    def _clear_stale_visuals(self, video_dir: Path) -> None:
        """Delete visual artifacts this run has decided not to reuse.

        Refusing to reuse is not enough on its own. A run only overwrites the
        artifacts it actually regenerates, so anything the previous backend left
        behind for a segment this run finishes differently — or never reaches —
        survives. The directory is then stamped with THIS backend's name, and
        the next run sees a matching marker, trusts the whole directory, and
        reuses the leftovers. Provenance is per-directory; contents are not,
        unless the mismatch is cleaned up at the point it is detected.

        Found in Ubu Rex e03: a killed Veo run left three 10KB Pillow stills
        from an earlier pass inside a directory marked `veo`, and segment 2 had
        only the still. A resume would have muxed it into the episode.

        Visuals only. The audio (`segment_NN.mp3`, `segment_NN_turns/`) is a
        different stage's output, costs real money to resynthesize, and has
        nothing to do with which visual backend ran.
        """
        stale = [
            p for p in video_dir.glob("segment_*")
            if p.is_file() and p.suffix in {".mp4", ".png"}
        ]
        if not stale:
            return
        for path in stale:
            path.unlink()
        # The flagged record named those files. Keeping it would carry verdicts
        # forward onto whatever regenerates into the same filenames.
        self._flagged.clear()
        (video_dir / FLAGGED_FILE).unlink(missing_ok=True)
        logger.info(
            "Cleared %d stale visual artifact(s) from %s so a later run cannot "
            "mistake them for '%s' output",
            len(stale), video_dir.name, self._backend.name,
        )

    def _record_provenance(self, video_dir: Path) -> None:
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / PROVENANCE_FILE).write_text(self._backend.name)

    def _cached(self, output_stem: Path, width: int, height: int) -> VisualOutput | None:
        """An already-generated asset for this output stem, or None to generate.

        Backends write `<stem>.mp4` (Veo) or `<stem>.png` (Pillow, and Veo's
        safety-filter fallback), so both are candidates.

        A video is only accepted if ffprobe can read a duration from it. A run
        killed mid-download leaves a plausible-looking but truncated MP4, and
        silently reusing that would produce a broken episode far downstream —
        much more expensive to diagnose than the regeneration it saves.
        """
        if not self._reuse_this_run:
            return None

        for suffix, out_type in (
            (".mp4", VisualOutputType.VIDEO),
            (".png", VisualOutputType.IMAGE),
        ):
            path = output_stem.with_suffix(suffix)
            if not path.exists() or path.stat().st_size == 0:
                continue
            if path.name in self._flagged:
                logger.warning(
                    "Existing clip %s failed continuity on its last run — "
                    "regenerating rather than reusing it",
                    path.name,
                )
                continue
            if out_type == VisualOutputType.VIDEO and _probe_duration(path) <= 0:
                logger.warning(
                    "Existing clip %s is unreadable (truncated?) — regenerating",
                    path.name,
                )
                continue
            return VisualOutput(
                path=path, output_type=out_type, width=width, height=height
            )
        return None

    async def _generate_reviewed(
        self,
        segment: Segment,
        output_path: Path,
        width: int,
        height: int,
        config: dict[str, Any],
        label: str,
        reference_path: str,
        work_dir: Path,
        expected_characters: str,
        reviews: list[dict[str, Any]],
        intended_action: str = "",
    ) -> VisualOutput:
        """Generate one clip, review it, and retry on a fatal continuity failure.

        Reviewing INLINE rather than in a pass at the end is the whole point: a
        bad clip becomes the chain anchor for the next one, so a corrupted clip 1
        silently poisons clips 2 and 3. Catching it here means the rest of the
        segment never inherits it.

        Escalation, mirroring the safety-filter ladder already in this stage:
          1. Generate as configured.
          2. On a fatal verdict, regenerate unchanged — the failures are
             statistical, and the same inputs often succeed on a second roll.
          3. Still fatal: drop `chain_from` and re-anchor to the hero reference.
             When the previous clip's last frame is what went wrong, chaining
             from it again just reproduces the fault.
        Exhausted: keep the last attempt and record it. A flagged clip in a
        finished episode beats a failed render, but it must be reported.
        """
        attempts: list[VisualOutput] = []
        verdict = None

        for attempt in range(1, CRITIC_MAX_ATTEMPTS + 1):
            attempt_config = dict(config)
            escalated = attempt >= 3 and config.get("chain_from")
            if escalated:
                # Re-anchor to the hero instead of the (suspect) chained frame.
                attempt_config.pop("chain_from", None)

            result = await self._backend.generate(
                segment=segment,
                output_path=output_path,
                width=width,
                height=height,
                config=attempt_config,
            )
            attempts.append(result)

            if result.output_type != VisualOutputType.VIDEO:
                # A still means the backend already fell back (safety filter);
                # its own retry ladder handled that. Nothing for the critic to do.
                return result
            if self._critic is None or not self._critic.available:
                return result

            verdict = self._critic.review(
                clip_path=result.path,
                reference_path=Path(reference_path),
                work_dir=work_dir,
                expected_characters=expected_characters,
                # Scoped to THIS clip. The whole segment's directions would tell
                # the critic to expect a one-time action in every clip, which is
                # how e03 passed four consecutive exits as intended.
                intended_action=intended_action,
            )
            reviews.append(
                {
                    "clip": result.path.name,
                    "attempt": attempt,
                    "passed": verdict.passed,
                    "unverified": verdict.unverified,
                    "detail": verdict.describe(),
                }
            )

            if verdict.passed:
                if verdict.unverified:
                    logger.warning("%s: NOT VERIFIED — %s", label, verdict.summary)
                else:
                    logger.info("%s: critic ok — %s", label, verdict.summary)
                # A clip that now passes is no longer flagged. Without this the
                # record outlives the problem and every future run regenerates a
                # clip that has already been fixed.
                if result.path.name in self._flagged:
                    self._flagged.discard(result.path.name)
                    self._save_flagged()
                return result

            logger.warning(
                "%s: critic REJECTED (attempt %d/%d) — %s",
                label, attempt, CRITIC_MAX_ATTEMPTS, verdict.describe(),
            )
            if attempt < CRITIC_MAX_ATTEMPTS:
                logger.info(
                    "%s: regenerating%s",
                    label,
                    " without chain anchor (re-anchoring to hero)"
                    if attempt + 1 >= 3 and config.get("chain_from")
                    else "",
                )

        logger.error(
            "%s: still failing continuity after %d attempts — keeping last clip. "
            "FLAGGED FOR REVIEW: %s",
            label, CRITIC_MAX_ATTEMPTS,
            verdict.describe() if verdict else "unknown",
        )
        # Record it so a resume regenerates rather than adopting it. Reuse never
        # re-reviews, so an inherited flagged clip is never looked at again.
        self._flagged.add(attempts[-1].path.name)
        self._save_flagged()
        return attempts[-1]

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        file_store = FileStore()
        vis_cfg = ctx.settings.visuals
        vid_cfg = ctx.settings.video

        # Merge default visuals config with strategy overrides
        config: dict[str, Any] = {
            "font_size": vis_cfg.font_size,
            "font_color": vis_cfg.font_color,
            "text_margin": vis_cfg.text_margin,
            "gradient_top": vis_cfg.gradient_top,
            "gradient_bottom": vis_cfg.gradient_bottom,
        }
        config.update(ctx.strategy.visuals)

        # Resolve the optional hero-variant manifest. Strategies that declare
        # `visuals.variants_manifest` use per-segment reference images chosen
        # by VariantSelectionStage. Strategies without it use the existing
        # `reference_image` field unchanged.
        variant_resolver = _build_variant_resolver(ctx.strategy.visuals)
        speaker_descriptions = _speaker_descriptions(ctx.strategy.visuals)

        has_video_clips = False
        # Per-segment clip lists for assembly. Only populated when a video-output
        # segment needed >1 clip; assembly falls back to seg.image_path otherwise.
        segment_clips: dict[int, list[str]] = ctx.artifacts.setdefault(
            "segment_clips", {}
        )
        # Critic findings, surfaced after the run so an unattended batch reports
        # what it flagged instead of burying it in the log.
        critic_reviews: list[dict[str, Any]] = ctx.artifacts.setdefault(
            "critic_reviews", []
        )

        # Provenance gates clip reuse for the whole run, so decide once.
        run_dir = file_store.video_dir(ctx.video.id)
        self._run_dir = run_dir
        self._flagged = self._load_flagged(run_dir)
        if self._flagged:
            logger.info(
                "%d clip(s) flagged by a previous run will be regenerated: %s",
                len(self._flagged), ", ".join(sorted(self._flagged)),
            )
        self._reuse_this_run = self._reuse_allowed(run_dir)
        if not self._reuse_this_run:
            # Clear before stamping. Anything left behind would sit in a
            # directory about to claim this backend's name, and the next run
            # would reuse it on the strength of that claim.
            self._clear_stale_visuals(run_dir)
        self._record_provenance(run_dir)

        for seg in ctx.script.segments:
            video_dir = file_store.video_dir(ctx.video.id)
            # Per-segment config: same as global config, but override
            # reference_image based on the segment's selected hero variant.
            seg_config = dict(config)
            _apply_camera_move(seg_config, seg.index)
            # Clip 0 covers the first window of the segment's audio; chained
            # clips get their own window below.
            seg_config["speech_schedule"] = build_speech_schedule(
                seg, 0, CLIP_TARGET_SECONDS, speaker_descriptions
            )
            # Physical descriptions of who should be on screen, so the critic
            # knows what it's checking for rather than inferring from the image.
            expected_characters = ", ".join(
                speaker_descriptions.get(normalize_speaker(s), s)
                for s in seg.speakers
            )
            resolved_ref = variant_resolver(seg.hero_variant)
            if resolved_ref:
                seg_config["reference_image"] = resolved_ref
                logger.info(
                    "Segment %d hero variant '%s' → %s",
                    seg.index, seg.hero_variant or "(default)",
                    Path(resolved_ref).name,
                )

            # Known before clip 0 because it depends only on the audio length,
            # and clip 0 needs it to scope its own stage directions.
            target_seconds = seg.actual_duration or seg.estimated_duration
            n_clips_total = max(1, math.ceil(target_seconds / CLIP_TARGET_SECONDS))

            first_action = build_staged_action(
                seg, 0, CLIP_TARGET_SECONDS, n_clips_total
            )
            seg_config["visual_prompt_override"] = scope_visual_prompt(
                seg.visual_prompt, first_action
            )

            # Always generate clip 0 first so we can see the output type before
            # deciding whether to generate more.
            first_output = video_dir / f"segment_{seg.index:02d}"
            first_result = self._cached(first_output, vid_cfg.width, vid_cfg.height)
            if first_result is not None:
                logger.info(
                    "Segment %d clip 1: reusing existing %s",
                    seg.index, first_result.path.name,
                )
            else:
                logger.info(
                    "Generating visual for segment %d [%s] (clip 1)",
                    seg.index, self._backend.name,
                )
                first_result = await self._generate_reviewed(
                    segment=seg,
                    output_path=first_output,
                    width=vid_cfg.width,
                    height=vid_cfg.height,
                    config=seg_config,
                    label=f"Segment {seg.index} clip 1",
                    reference_path=seg_config.get("reference_image", ""),
                    work_dir=video_dir / "critic",
                    expected_characters=expected_characters,
                    reviews=critic_reviews,
                    intended_action=first_action,
                )
            seg.image_path = str(first_result.path)
            segment_types = ctx.artifacts.setdefault("segment_visual_types", {})
            segment_types[seg.index] = first_result.output_type

            if first_result.output_type == VisualOutputType.VIDEO:
                has_video_clips = True

            # Multi-clip path: if the backend produces video AND the audio is
            # longer than one Veo clip, generate additional clips to cover it.
            # Within a segment we CHAIN clips by extracting the last frame of
            # clip M and passing it as the starting image for clip M+1 (via
            # `chain_from` in config). This makes sub-clip cuts within a
            # segment continuous — Bartholomew's pose, lighting, and motion
            # pick up exactly where they left off. The first clip of each
            # segment re-anchors to the hero reference image so the character
            # doesn't drift across segments.
            clip_paths: list[str] = [str(first_result.path)]
            if first_result.output_type == VisualOutputType.VIDEO:
                if n_clips_total > 1:
                    logger.info(
                        "Segment %d needs %d clips for %.1fs audio (chained)",
                        seg.index, n_clips_total, target_seconds,
                    )
                for extra_idx in range(1, n_clips_total):
                    # Extract last frame of the previous clip, pass it as the
                    # chain anchor for this clip.
                    prev_clip = Path(clip_paths[-1])
                    last_frame = video_dir / (
                        f"segment_{seg.index:02d}_clip_{extra_idx - 1:02d}_lastframe.png"
                        if extra_idx > 1
                        else f"segment_{seg.index:02d}_lastframe.png"
                    )
                    _extract_last_frame(prev_clip, last_frame)
                    chain_config = {
                        **seg_config,
                        "chain_from": str(last_frame),
                        # This clip covers a later slice of the audio, so it
                        # needs its own dialogue window — reusing clip 0's
                        # would tell the model the wrong puppet is speaking.
                        "speech_schedule": build_speech_schedule(
                            seg, extra_idx, CLIP_TARGET_SECONDS, speaker_descriptions
                        ),
                    }
                    # Same reasoning as the schedule: the directions belong to
                    # the turns this clip covers, not to the whole segment.
                    extra_action = build_staged_action(
                        seg, extra_idx, CLIP_TARGET_SECONDS, n_clips_total
                    )
                    chain_config["visual_prompt_override"] = scope_visual_prompt(
                        seg.visual_prompt, extra_action
                    )

                    extra_output = (
                        video_dir / f"segment_{seg.index:02d}_clip_{extra_idx:02d}"
                    )
                    cached = self._cached(extra_output, vid_cfg.width, vid_cfg.height)
                    if cached is not None:
                        logger.info(
                            "Segment %d clip %d/%d: reusing existing %s",
                            seg.index, extra_idx + 1, n_clips_total, cached.path.name,
                        )
                        clip_paths.append(str(cached.path))
                        continue
                    logger.info(
                        "Generating visual for segment %d [%s] (clip %d/%d, chained)",
                        seg.index, self._backend.name, extra_idx + 1, n_clips_total,
                    )
                    extra_result = await self._generate_reviewed(
                        segment=seg,
                        output_path=extra_output,
                        width=vid_cfg.width,
                        height=vid_cfg.height,
                        config=chain_config,
                        label=f"Segment {seg.index} clip {extra_idx + 1}",
                        # Judge chained clips against the SEGMENT'S HERO, not the
                        # frame they chained from — otherwise a drifting clip is
                        # compared against the drift and passes.
                        reference_path=seg_config.get("reference_image", ""),
                        work_dir=video_dir / "critic",
                        expected_characters=expected_characters,
                        reviews=critic_reviews,
                        intended_action=extra_action,
                    )

                    # If the chained generation got rejected (e.g., Veo safety
                    # filter on the chained frame) and the backend fell back
                    # to a still image, retry once anchored to the hero ref
                    # instead — chained frames are sometimes darker/more
                    # skeletal in ways that trigger filters the clean hero
                    # doesn't. If that *also* fails, we stop multi-clip
                    # generation for this segment rather than mixing video
                    # and still-image paths through the rest of the pipeline.
                    if extra_result.output_type != VisualOutputType.VIDEO:
                        logger.warning(
                            "Segment %d clip %d chained gen produced %s (safety filter?); "
                            "retrying with hero-ref anchor",
                            seg.index, extra_idx, extra_result.output_type.value,
                        )
                        extra_result = await self._backend.generate(
                            segment=seg,
                            output_path=extra_output,
                            width=vid_cfg.width,
                            height=vid_cfg.height,
                            config=seg_config,  # no chain_from → falls back to per-segment hero
                        )
                        if extra_result.output_type != VisualOutputType.VIDEO:
                            logger.warning(
                                "Segment %d clip %d hero-ref retry also failed; "
                                "stopping multi-clip gen with %d video clip(s). "
                                "Final muxed clip will be %.1fs short of audio.",
                                seg.index, extra_idx, len(clip_paths),
                                target_seconds - len(clip_paths) * CLIP_TARGET_SECONDS,
                            )
                            break
                    clip_paths.append(str(extra_result.path))

            if len(clip_paths) > 1:
                segment_clips[seg.index] = clip_paths
                logger.info(
                    "Segment %d: %d clips generated", seg.index, len(clip_paths),
                )

            logger.info(
                "Visual saved: %s (%s)",
                first_result.path.name, first_result.output_type.value,
            )

        # Tell assembly whether it's dealing with stills or pre-animated clips
        ctx.artifacts["visual_output_type"] = (
            VisualOutputType.VIDEO if has_video_clips else VisualOutputType.IMAGE
        )
        ctx.video.status = VideoStatus.VISUALS_DONE

        total_clips = sum(
            len(segment_clips.get(s.index, [s.image_path]))
            for s in ctx.script.segments
        )
        logger.info(
            "Visual generation complete: %d clips across %d segments via %s",
            total_clips, len(ctx.script.segments), self._backend.name,
        )
        _log_critic_summary(critic_reviews)
        return ctx


def _log_critic_summary(reviews: list[dict[str, Any]]) -> None:
    """Report what the critic found, at the end where it can't be missed.

    The point of an unattended run is that nobody is watching the log scroll by.
    A clip that failed every attempt, or a run where the critic never actually
    ran, has to be visible in the last few lines — otherwise "it finished" gets
    read as "it's fine".
    """
    if not reviews:
        return

    clips = {r["clip"] for r in reviews}
    unverified = {r["clip"] for r in reviews if r["unverified"]}
    regenerated = {r["clip"] for r in reviews if not r["passed"]}
    # A clip is only really failing if its LAST attempt failed; earlier
    # rejections that a retry fixed are a success story, not a problem.
    final = {r["clip"]: r for r in reviews}
    still_failing = [r for r in final.values() if not r["passed"]]

    logger.info(
        "Critic: reviewed %d clip(s), %d regenerated, %d unverified",
        len(clips), len(regenerated - unverified), len(unverified),
    )
    if unverified:
        logger.warning(
            "Critic could NOT verify %d clip(s) — these were not checked: %s",
            len(unverified), ", ".join(sorted(unverified)),
        )
    for r in still_failing:
        logger.error(
            "FLAGGED FOR REVIEW — %s failed continuity after %d attempt(s): %s",
            r["clip"], r["attempt"], r["detail"],
        )


def _probe_duration(path: Path) -> float:
    """Seconds of media at `path`, or 0.0 if it isn't readable."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


_ACTION_CLAUSE = re.compile(r"\s*Action:\s.*$", re.DOTALL)


def build_staged_action(
    segment: Segment, clip_index: int, clip_seconds: float, n_clips: int
) -> str:
    """The stage directions belonging to the slice of audio one clip covers.

    A stage direction annotates ONE turn and describes something that happens
    once. A segment, though, is several chained clips, so handing the whole
    segment's directions to every clip tells the model to perform a one-time
    action repeatedly — and because clip M+1 chains from clip M's last frame,
    it starts from the post-action state and has to reset in order to do it
    again. Ubu Rex e03 segment 1 carried "PERE UBU going out, slamming the
    door" into all four clips: he exited four times, and the door Veo invented
    to satisfy the direction was baked into every chain anchor until it
    dominated the frame.

    Scoped by `turn_timings`, the same windowing `build_speech_schedule` uses.
    Without timings a direction cannot be located, so it goes to the final clip
    — an unplaceable direction is far more likely to close a beat than open it,
    and one clip performing it is right where every clip performing it is wrong.
    """
    directions = [
        (t.speaker, t.stage_direction) for t in segment.turns if t.stage_direction.strip()
    ]
    if not directions:
        return ""

    if not segment.turn_timings:
        return segment.staged_action if clip_index == n_clips - 1 else ""

    window_start = clip_index * clip_seconds
    window_end = window_start + clip_seconds

    # Timings are per turn and in turn order, so zip against the turns to find
    # which window each annotated turn lands in.
    beats: list[str] = []
    for turn, timing in zip(segment.turns, segment.turn_timings):
        if not turn.stage_direction.strip():
            continue
        turn_end = timing.start + timing.duration
        if turn_end <= window_start or timing.start >= window_end:
            continue
        beats.append(f"{turn.speaker} {turn.stage_direction}".strip())

    return "; ".join(beats)


def scope_visual_prompt(visual_prompt: str, clip_action: str) -> str:
    """Replace the prompt's `Action:` clause with this clip's directions.

    `adapt_play.py` composes the whole segment's directions into a trailing
    `Action: ...` sentence, which then reaches every clip. Rewriting it per clip
    is what stops the model from staging the same exit four times.
    """
    base = _ACTION_CLAUSE.sub("", visual_prompt).strip()
    if not clip_action:
        return base
    return f"{base} Action: {clip_action}."


def build_speech_schedule(
    segment: Segment,
    clip_index: int,
    clip_seconds: float,
    descriptions: dict[str, str],
) -> str:
    """Timed 'who speaks when' for the slice of audio one clip covers.

    Image-to-video models never see the audio track, so without this they have
    no idea anyone is talking and animate every character's mouth at once —
    which reads as far more broken than imperfect lip-sync does. Telling the
    model that exactly one puppet speaks in each window, and that the others
    hold their mouths shut, recovers most of the perceived sync.

    Times are relative to this clip's start and clamped to its length. Speakers
    are named by physical description (from strategy.visuals.speaker_descriptions)
    because the model knows nothing about a name like "PERE UBU".

    Returns "" for single-narrator segments — narration has no speaker to show.
    """
    if not segment.turn_timings:
        return ""

    window_start = clip_index * clip_seconds
    window_end = window_start + clip_seconds

    beats: list[str] = []
    for timing in segment.turn_timings:
        turn_end = timing.start + timing.duration
        if turn_end <= window_start or timing.start >= window_end:
            continue
        rel_start = max(0.0, timing.start - window_start)
        rel_end = min(clip_seconds, turn_end - window_start)
        who = descriptions.get(normalize_speaker(timing.speaker), timing.speaker)
        beats.append(f"from {rel_start:.1f}s to {rel_end:.1f}s {who} is speaking")

    if not beats:
        return ""

    return (
        "Dialogue timing for this shot: "
        + "; ".join(beats)
        + ". Only the character described as speaking moves their mouth. Every "
        "other puppet keeps its mouth firmly closed and listens, reacting with "
        "small head turns and body movements only. Never move two mouths at once."
    )


def _speaker_descriptions(strategy_visuals: dict[str, Any]) -> dict[str, str]:
    """Normalized speaker-key -> physical description map."""
    raw = strategy_visuals.get("speaker_descriptions") or {}
    return {normalize_speaker(k): str(v) for k, v in raw.items()}


def _apply_camera_move(seg_config: dict[str, Any], segment_index: int) -> None:
    """Prepend a per-segment camera move onto the base animation_style.

    Cycles through the strategy's `camera_moves` list so every clip isn't an
    identical push-in. The base `animation_style` carries the look/medium; the
    move is prepended. No-op when the strategy declares no camera_moves. All
    chained sub-clips of a segment reuse the same seg_config, so the move is
    consistent within a segment and only varies between segments.
    """
    moves = seg_config.get("camera_moves") or []
    if not moves:
        return
    move = moves[segment_index % len(moves)]
    base_style = seg_config.get("animation_style", "")
    seg_config["animation_style"] = f"{move}, {base_style}" if base_style else move


def _build_variant_resolver(strategy_visuals: dict[str, Any]):
    """Return a fn(variant_key) -> reference_image_path-or-None.

    If the strategy declares a `variants_manifest`, the resolver maps a
    segment's `hero_variant` key to the absolute file path of the matching
    PNG. Falls back to the strategy's `reference_image` (singular) when the
    key isn't set or when the manifest isn't configured — preserving the
    pre-variants behavior for other strategies.
    """
    manifest_rel = strategy_visuals.get("variants_manifest")
    default_key = strategy_visuals.get("default_variant", "")
    fallback_ref = strategy_visuals.get("reference_image", "")

    variants_by_key: dict[str, str] = {}
    manifest_dir: Path | None = None
    if manifest_rel:
        manifest_path = PROJECT_ROOT / manifest_rel
        if manifest_path.exists():
            manifest = yaml.safe_load(manifest_path.read_text()) or {}
            manifest_dir = manifest_path.parent
            for v in manifest.get("variants", []):
                variants_by_key[v["key"]] = v["file"]
        else:
            logger.warning("variants_manifest %s not found", manifest_path)

    def resolve(variant_key: str) -> str:
        # Prefer the per-segment variant if it maps to a real file
        candidate_key = variant_key or default_key
        if candidate_key and candidate_key in variants_by_key and manifest_dir:
            return str(manifest_dir / variants_by_key[candidate_key])
        # Fall back to the strategy's singular reference_image (legacy path)
        return fallback_ref

    return resolve


def _extract_last_frame(video_path: Path, output_path: Path) -> None:
    """Extract the final frame of a video as a PNG for Veo chain anchoring.

    Uses -sseof to seek a tiny bit before EOF, then writes one frame.
    -update 1 + -frames:v 1 ensures a single-image output. -q:v 1 keeps
    quality high since this PNG becomes the starting frame for the next
    Veo clip and we want it to look exactly like the moment we left off.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-sseof", "-0.1",
        "-i", str(video_path),
        "-update", "1",
        "-frames:v", "1",
        "-q:v", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Last-frame extract failed for {video_path.name}: {result.stderr}"
        )
