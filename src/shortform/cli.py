"""CLI entry point — typer-based command interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from shortform.config import (
    PROJECT_ROOT,
    AppSettings,
    StrategyConfig,
    list_strategies,
    load_settings,
    load_strategy,
)
from shortform.models.script import Script
from shortform.models.video import Video
from shortform.pipeline.batch import (
    EpisodeResult,
    EpisodeStatus,
    critic_findings,
    reusable_output,
    run_batch,
    write_manifest,
)
from shortform.pipeline.context import PipelineContext
from shortform.pipeline.runner import PipelineRunner
from shortform.pipeline.stage import PipelineStage
from shortform.stages.assembly import AssemblyStage
from shortform.stages.script_gen import ScriptGenStage
from shortform.stages.tts import TTSStage
from shortform.stages.variant_select import VariantSelectionStage
from shortform.stages.visual_gen import VisualGenStage
from shortform.store.db import Database
from shortform.visuals import get_backend, list_backends
from shortform.visuals.backend import VisualBackend
from shortform.visuals.critic import ClipCritic

app = typer.Typer(
    name="shortform",
    help="Automated short-form video content creation.",
    no_args_is_help=True,
)


def _build_critic(
    settings: AppSettings, strategy: StrategyConfig, enabled: bool
) -> ClipCritic | None:
    """Continuity critic for generated clips, or None when disabled.

    Opt-in per strategy (`visuals.critic: true`) so existing strategies don't
    start paying for vision calls on an upgrade. Costs one vision call per
    generated clip — negligible against the Veo clip it may save regenerating.
    """
    if not enabled or not strategy.visuals.get("critic", False):
        return None
    import os

    key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        typer.echo(
            "Warning: visuals.critic is enabled but ANTHROPIC_API_KEY is not set "
            "— clips will NOT be checked for continuity.",
            err=True,
        )
        return None
    from shortform.visuals.critic import DEFAULT_MODEL

    return ClipCritic(
        api_key=key,
        model=strategy.visuals.get("critic_model", DEFAULT_MODEL),
    )


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def generate(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name to use"),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Override topic selection"),
    visual_backend: str | None = typer.Option(
        None, "--visual-backend", "-vb", help="Visual backend (pillow, veo)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Generate a short-form video from a strategy."""
    _setup_logging(verbose)
    logger = logging.getLogger("shortform.cli")

    # Load config
    settings = load_settings()
    if not settings.anthropic_api_key:
        # Try bare env var
        import os

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            settings.anthropic_api_key = key
        else:
            typer.echo("Error: ANTHROPIC_API_KEY not set. See .env.example", err=True)
            raise typer.Exit(1)

    # Load strategy
    try:
        strat = load_strategy(strategy)
    except FileNotFoundError:
        typer.echo(f"Error: Strategy '{strategy}' not found.", err=True)
        typer.echo(f"Available: {', '.join(list_strategies())}", err=True)
        raise typer.Exit(1)

    # Initialize DB
    paths = settings.paths.resolve()
    db = Database(paths["db_path"])
    db.initialize()

    # Resolve visual backend: CLI flag > strategy declaration > global default.
    # Letting the strategy declare its backend stops the flagship (gothic) from
    # silently falling back to Pillow gradients when no flag is passed.
    backend_name = visual_backend or strat.visuals.get("backend") or settings.visuals.backend
    try:
        backend_kwargs: dict[str, str] = {}
        if backend_name == "veo":
            import os

            gemini_key = settings.google_gemini_api_key
            if not gemini_key:
                gemini_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
            if gemini_key:
                backend_kwargs["api_key"] = gemini_key
        backend = get_backend(backend_name, **backend_kwargs)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(f"Available backends: {', '.join(list_backends())}", err=True)
        raise typer.Exit(1)

    logger.info("Using visual backend: %s", backend.name)

    # Build pipeline
    stages: list[PipelineStage] = [
        ScriptGenStage(),
        VariantSelectionStage(),
        TTSStage(),
        VisualGenStage(backend=backend),
        AssemblyStage(),
    ]
    runner = PipelineRunner(stages=stages, db=db)

    # Create context
    video = Video(strategy_name=strategy)
    ctx = PipelineContext(
        settings=settings,
        strategy=strat,
        video=video,
        topic=topic or "",
    )

    db.save_video(video)
    logger.info("Starting pipeline for strategy '%s' (video: %s)", strategy, video.id)

    # Run pipeline
    ctx = asyncio.run(runner.run(ctx))

    if ctx.errors:
        typer.echo(f"\nPipeline failed: {ctx.errors[-1]}", err=True)
        raise typer.Exit(1)

    typer.echo("\nVideo generated successfully!")
    typer.echo(f"  ID:       {ctx.video.id}")
    typer.echo(f"  Title:    {ctx.video.title}")
    typer.echo(f"  Duration: {ctx.video.duration:.1f}s")
    typer.echo(f"  Output:   {ctx.video.output_path}")
    typer.echo(f"  Size:     {ctx.video.file_size_bytes / (1024 * 1024):.1f} MB")


@app.command()
def script(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name to use"),
    topic: str | None = typer.Option(None, "--topic", "-t", help="Optional topic override"),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="Where to write the script JSON (default: data/scripts/<id>.json)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Generate a script only — no TTS, no visuals, no assembly.

    Writes the script JSON to disk so you can review/edit before committing
    to the expensive stages. Run `generate-from-script <path>` to pick up
    where this left off.
    """
    _setup_logging(verbose)

    settings = load_settings()
    if not settings.anthropic_api_key:
        import os
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            settings.anthropic_api_key = key
        else:
            typer.echo("Error: ANTHROPIC_API_KEY not set. See .env.example", err=True)
            raise typer.Exit(1)

    try:
        strat = load_strategy(strategy)
    except FileNotFoundError:
        typer.echo(f"Error: Strategy '{strategy}' not found.", err=True)
        typer.echo(f"Available: {', '.join(list_strategies())}", err=True)
        raise typer.Exit(1)

    ctx = PipelineContext(
        settings=settings,
        strategy=strat,
        video=Video(strategy_name=strategy),
        topic=topic or "",
    )

    asyncio.run(ScriptGenStage().execute(ctx))

    out_path = output or (PROJECT_ROOT / "data" / "scripts" / f"{ctx.script.id}.json")
    ctx.script.save_json(out_path)

    typer.echo(f"\nScript saved: {out_path}")
    typer.echo(f"  ID:       {ctx.script.id}")
    typer.echo(f"  Title:    {ctx.script.title}")
    typer.echo(f"  Topic:    {ctx.script.topic}")
    est_dur = ctx.script.total_duration
    typer.echo(f"  Segments: {ctx.script.segment_count} (~{est_dur:.1f}s estimated)")
    typer.echo("")
    for seg in ctx.script.segments:
        typer.echo(f"  [{seg.index}] ({seg.estimated_duration:.1f}s) {seg.narration}")
        if seg.visual_prompt:
            vp = seg.visual_prompt
            ellipsis = "..." if len(vp) > 90 else ""
            typer.echo(f"      visual: {vp[:90]}{ellipsis}")
    typer.echo("")
    typer.echo(f"Next: uv run shortform generate-from-script '{out_path}'")


@app.command("generate-from-script")
def generate_from_script(
    script_path: Path = typer.Argument(..., help="Path to a script JSON file"),
    visual_backend: str | None = typer.Option(
        None, "--visual-backend", "-vb", help="Visual backend (pillow, veo)"
    ),
    video_id: str | None = typer.Option(
        None, "--video-id",
        help="Resume into an existing video's working directory, reusing any "
             "clips already generated there. Use after an interrupted run so "
             "you don't pay to regenerate what succeeded.",
    ),
    regenerate: bool = typer.Option(
        False, "--regenerate",
        help="Regenerate visuals even when clips already exist (use after "
             "changing prompts, reference images, or camera config).",
    ),
    no_critic: bool = typer.Option(
        False, "--no-critic",
        help="Skip the continuity critic even if the strategy enables it.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run TTS → visuals → assembly from an existing script JSON.

    Skips script_gen entirely (no Claude call, no ANTHROPIC key needed for
    that stage). The script's strategy_name determines which strategy YAML
    is loaded for TTS/visuals/music config.

    Visual generation reuses clips already present in the working directory, so
    an interrupted run resumes with `--video-id <id>` and only pays for what's
    missing. Pass `--regenerate` to force fresh visuals.
    """
    _setup_logging(verbose)
    logger = logging.getLogger("shortform.cli")

    if not script_path.exists():
        typer.echo(f"Error: script file not found: {script_path}", err=True)
        raise typer.Exit(1)

    script_obj = Script.load_json(script_path)
    if not script_obj.strategy_name:
        typer.echo(
            "Error: script JSON has no strategy_name — can't determine which "
            "strategy to load for TTS/visuals.", err=True,
        )
        raise typer.Exit(1)

    settings = load_settings()
    try:
        strat = load_strategy(script_obj.strategy_name)
    except FileNotFoundError:
        typer.echo(
            f"Error: Strategy '{script_obj.strategy_name}' (from script JSON) not found.",
            err=True,
        )
        typer.echo(f"Available: {', '.join(list_strategies())}", err=True)
        raise typer.Exit(1)

    paths = settings.paths.resolve()
    db = Database(paths["db_path"])
    db.initialize()

    backend_name = visual_backend or strat.visuals.get("backend") or settings.visuals.backend
    try:
        backend_kwargs: dict[str, str] = {}
        if backend_name == "veo":
            import os
            gemini_key = settings.google_gemini_api_key
            if not gemini_key:
                gemini_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
            if gemini_key:
                backend_kwargs["api_key"] = gemini_key
        backend = get_backend(backend_name, **backend_kwargs)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(f"Available backends: {', '.join(list_backends())}", err=True)
        raise typer.Exit(1)

    logger.info("Using visual backend: %s", backend.name)

    stages: list[PipelineStage] = [
        ScriptGenStage(),
        VariantSelectionStage(),
        TTSStage(),
        VisualGenStage(
            backend=backend,
            reuse_existing=not regenerate,
            critic=_build_critic(settings, strat, enabled=not no_critic),
        ),
        AssemblyStage(),
    ]
    runner = PipelineRunner(stages=stages, db=db)

    # Reusing a video id means reusing its working directory, which is what makes
    # already-generated clips discoverable. A fresh id gets a fresh directory and
    # therefore regenerates everything.
    video = Video(
        strategy_name=script_obj.strategy_name,
        topic=script_obj.topic,
        title=script_obj.title,
        script_id=script_obj.id,
    )
    if video_id:
        video.id = video_id
        logger.info("Resuming into existing video %s — reusing any clips found", video_id)
    ctx = PipelineContext(
        settings=settings,
        strategy=strat,
        video=video,
        script=script_obj,
        topic=script_obj.topic,
    )

    db.save_video(video)
    logger.info(
        "Resuming pipeline from script JSON (%s) — skipping script_gen. video=%s",
        script_path.name, video.id,
    )

    # Reference the stage's own name rather than a bare literal so a rename of
    # ScriptGenStage.name can't silently turn this into a no-op (the runner now
    # also fails loudly if resume_from matches no stage).
    ctx = asyncio.run(runner.run(ctx, resume_from=stages[0].name))

    if ctx.errors:
        typer.echo(f"\nPipeline failed: {ctx.errors[-1]}", err=True)
        raise typer.Exit(1)

    typer.echo("\nVideo generated successfully!")
    typer.echo(f"  ID:       {ctx.video.id}")
    typer.echo(f"  Title:    {ctx.video.title}")
    typer.echo(f"  Duration: {ctx.video.duration:.1f}s")
    typer.echo(f"  Output:   {ctx.video.output_path}")
    typer.echo(f"  Size:     {ctx.video.file_size_bytes / (1024 * 1024):.1f} MB")


@app.command("list-strategies")
def list_strats() -> None:
    """List available content strategies."""
    strategies = list_strategies()
    if not strategies:
        typer.echo("No strategies found in config/strategies/")
        raise typer.Exit(1)
    typer.echo("Available strategies:")
    for name in strategies:
        strat = load_strategy(name)
        typer.echo(f"  {name}: {strat.description}")


@app.command("list-videos")
def list_videos(
    strategy: str | None = typer.Option(None, "--strategy", "-s"),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List generated videos."""
    settings = load_settings()
    paths = settings.paths.resolve()
    db = Database(paths["db_path"])
    db.initialize()

    videos = db.list_videos(strategy=strategy, limit=limit)
    if not videos:
        typer.echo("No videos found.")
        return

    typer.echo(f"{'ID':<14} {'Status':<12} {'Strategy':<20} {'Duration':>8}  Title")
    typer.echo("-" * 80)
    for v in videos:
        typer.echo(
            f"{v.id:<14} {v.status.value:<12} {v.strategy_name:<20} "
            f"{v.duration:>6.1f}s  {v.title[:30]}"
        )


@app.callback()
def main() -> None:
    """Automated short-form video content creation platform."""


if __name__ == "__main__":
    app()


def _resolve_backend_name(
    settings: AppSettings, strategy: StrategyConfig, override: str | None
) -> str:
    """Visual backend name from CLI flag > strategy declaration > global default."""
    return str(override or strategy.visuals.get("backend") or settings.visuals.backend)


def _resolve_visual_backend(
    settings: AppSettings, strategy: StrategyConfig, override: str | None
) -> VisualBackend:
    """Instantiate the resolved visual backend."""
    import os

    name = _resolve_backend_name(settings, strategy, override)
    kwargs: dict[str, str] = {}
    if name == "veo":
        key = settings.google_gemini_api_key or os.environ.get(
            "GOOGLE_GEMINI_API_KEY", ""
        )
        if key:
            kwargs["api_key"] = key
    return get_backend(name, **kwargs)


@app.command("batch")
def batch(
    script_paths: list[Path] = typer.Argument(
        ..., help="Script JSON files to render (shell globs work)."
    ),
    visual_backend: str | None = typer.Option(
        None, "--visual-backend", "-vb", help="Visual backend (pillow, veo)"
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Re-render episodes that already have a finished video, even when "
             "the recorded backend matches.",
    ),
    regenerate: bool = typer.Option(
        False, "--regenerate",
        help="Regenerate visuals even when clips already exist on disk.",
    ),
    no_critic: bool = typer.Option(
        False, "--no-critic", help="Skip the continuity critic."
    ),
    stop_on_failure: bool = typer.Option(
        False, "--stop-on-failure",
        help="Abort the batch on the first failure instead of continuing.",
    ),
    report_path: Path | None = typer.Option(
        None, "--report", help="Write a JSON report here."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would render without spending anything."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Render many episodes unattended, then report what needs a human.

    Each episode's video id is its script id, so assets are addressable
    (data/assets/<script_id>/) and a re-run after a crash reuses every clip the
    previous attempt already paid for. Finished episodes are skipped outright.

    A failure isolates to its own episode and the batch continues — except
    depleted Veo credits, which abort the run, since every remaining episode
    would fail the same way.

        uv run shortform batch data/scripts/uburex01e*.json -vb veo
        uv run shortform batch data/scripts/*.json --dry-run
    """
    _setup_logging(verbose)
    logger = logging.getLogger("shortform.cli")

    scripts = sorted({p for p in script_paths if p.suffix == ".json"})
    missing = [p for p in scripts if not p.exists()]
    if missing:
        typer.echo(f"Error: script(s) not found: {', '.join(map(str, missing))}", err=True)
        raise typer.Exit(1)
    if not scripts:
        typer.echo("Error: no script JSON files given.", err=True)
        raise typer.Exit(1)

    settings = load_settings()
    paths = settings.paths.resolve()
    videos_dir = paths["videos_dir"]

    def backend_for(script_path: Path) -> str:
        """Backend name this episode would render with, for the skip check.

        Resolved per-script because the strategy (read from the script JSON) can
        declare its own backend — the flagship declares `veo` so a bare batch
        doesn't silently fall back to Pillow gradients.
        """
        try:
            strategy_name = Script.load_json(script_path).strategy_name
            return _resolve_backend_name(
                settings, load_strategy(strategy_name), visual_backend
            )
        except Exception:  # noqa: BLE001 — render_one reports the real error
            return visual_backend or settings.visuals.backend

    def existing_output(script_path: Path) -> str:
        """A finished video for this episode, or "" if it needs rendering.

        Backend-aware: an output only counts as done if it was produced by the
        backend we're about to use. Otherwise a cheap `-vb pillow` test run
        would mark every episode finished and a later Veo pass would render
        nothing at all.
        """
        if force:
            return ""
        return reusable_output(videos_dir, script_path.stem, backend_for(script_path))

    if dry_run:
        typer.echo(f"Would process {len(scripts)} script(s):")
        # Evaluate once per script — existing_output logs, so calling it twice
        # would duplicate every warning.
        plan = [(p, existing_output(p), backend_for(p)) for p in scripts]
        for script_path, done, backend_name in plan:
            state = (
                f"skip (exists: {Path(done).name})" if done
                else f"RENDER via {backend_name}"
            )
            typer.echo(f"  [{state}] {script_path.stem}")
        pending = [p for p, done, _ in plan if not done]
        typer.echo("")
        typer.echo(f"{len(pending)} episode(s) would render. Nothing was spent.")
        return

    db = Database(paths["db_path"])
    db.initialize()

    async def render_one(script_path: Path) -> EpisodeResult:
        result = EpisodeResult(script_path=script_path, script_id=script_path.stem)
        script_obj = Script.load_json(script_path)
        if not script_obj.strategy_name:
            result.status = EpisodeStatus.FAILED
            result.error = "script JSON has no strategy_name"
            return result

        strat = load_strategy(script_obj.strategy_name)
        backend = _resolve_visual_backend(settings, strat, visual_backend)

        stages: list[PipelineStage] = [
            ScriptGenStage(),
            VariantSelectionStage(),
            TTSStage(),
            VisualGenStage(
                backend=backend,
                reuse_existing=not regenerate,
                critic=_build_critic(settings, strat, enabled=not no_critic),
            ),
            AssemblyStage(),
        ]
        runner = PipelineRunner(stages=stages, db=db)

        # The script id IS the video id — that is what makes assets addressable
        # and lets a re-run reuse clips from an interrupted attempt.
        video = Video(
            id=script_path.stem,
            strategy_name=script_obj.strategy_name,
            topic=script_obj.topic,
            title=script_obj.title,
            script_id=script_obj.id,
        )
        ctx = PipelineContext(
            settings=settings, strategy=strat, video=video,
            script=script_obj, topic=script_obj.topic,
        )
        db.save_video(video)

        ctx = await runner.run(ctx, resume_from=stages[0].name)

        result.title = ctx.video.title
        result.flagged, result.unverified = critic_findings(ctx.artifacts)
        if ctx.errors:
            result.status = EpisodeStatus.FAILED
            result.error = ctx.errors[-1]
        else:
            result.status = EpisodeStatus.COMPLETED
            result.duration = ctx.video.duration
            result.output_path = str(ctx.video.output_path)
            # Record which backend produced this, so a later batch can tell a
            # finished episode from one rendered with a different backend.
            write_manifest(videos_dir, result, backend.name)
        return result

    logger.info("Batch: %d script(s)", len(scripts))
    report = asyncio.run(
        run_batch(
            scripts=scripts,
            render=render_one,
            already_rendered=existing_output,
            stop_on_failure=stop_on_failure,
        )
    )

    typer.echo("")
    for line in report.summary_lines():
        typer.echo(line)

    if report_path:
        import json

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2))
        typer.echo("")
        typer.echo(f"Report written: {report_path}")

    if not report.ok:
        raise typer.Exit(1)


@app.command("publish-auth")
def publish_auth(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Authorize YouTube uploads. Run once; writes a refresh token to .env.

    Needs YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env, from a Google
    Cloud OAuth client of type "Desktop app" with the YouTube Data API v3
    enabled.
    """
    _setup_logging(verbose)
    from shortform.publish import oauth

    try:
        oauth.run_consent_flow()
    except oauth.MissingCredentials as e:
        typer.echo(f"Missing credentials: {e}")
        typer.echo(
            "\nAdd these to .env from your Google Cloud OAuth client:\n"
            "  YOUTUBE_CLIENT_ID=...\n"
            "  YOUTUBE_CLIENT_SECRET=..."
        )
        raise typer.Exit(1) from e
    except RuntimeError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from e

    typer.echo(f"\nSaved YOUTUBE_REFRESH_TOKEN to {oauth.ENV_FILE}")
    typer.echo("You can now run `shortform publish <video-id>`.")


@app.command()
def publish(
    video_id: str = typer.Argument(..., help="Script/video id, e.g. uburex01e01"),
    privacy: str = typer.Option(
        "private", "--privacy",
        help="private | unlisted | public. Defaults to private so nothing "
             "reaches an audience without a human step.",
    ),
    title: str | None = typer.Option(None, "--title", help="Override the title"),
    allow_flagged: bool = typer.Option(
        False, "--allow-flagged",
        help="Upload even though the critic flagged clips in this episode.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be uploaded, spend nothing."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Upload a rendered episode to YouTube.

    Deliberately separate from `generate` and `batch`: rendering is unattended,
    but publishing is the one irreversible step, so it stays a thing you run
    after watching the episode.
    """
    _setup_logging(verbose)
    from shortform.publish import episode as ep_mod
    from shortform.publish import oauth, youtube

    if privacy not in ("private", "unlisted", "public"):
        typer.echo(f"Invalid --privacy {privacy!r}: use private, unlisted or public")
        raise typer.Exit(1)

    try:
        ep = ep_mod.load_episode(video_id)
    except FileNotFoundError as e:
        typer.echo(str(e))
        raise typer.Exit(1) from e

    blockers = ep_mod.blocking_reasons(ep, allow_flagged=allow_flagged)
    for reason in blockers:
        typer.echo(f"BLOCKED: {reason}")
    if blockers:
        raise typer.Exit(1)

    metadata = youtube.build_metadata(
        title=title or ep.title,
        description=ep_mod.build_description(ep),
        tags=ep_mod.build_tags(ep),
        privacy=privacy,
        category_id=str(ep.publish_config.get("category_id", youtube.DEFAULT_CATEGORY_ID)),
    )

    size_mb = ep.video_path.stat().st_size / 1e6
    typer.echo(f"Episode:  {ep.video_id}")
    typer.echo(f"File:     {ep.video_path.name} ({size_mb:.1f} MB)")
    typer.echo(f"Title:    {metadata['snippet']['title']}")
    typer.echo(f"Privacy:  {privacy}")
    if metadata["snippet"]["tags"]:
        typer.echo(f"Tags:     {', '.join(metadata['snippet']['tags'])}")
    typer.echo("Description:")
    for line in metadata["snippet"]["description"].splitlines():
        typer.echo(f"  {line}")
    if ep.unverified:
        typer.echo(
            f"\nNote: {len(ep.unverified)} clip(s) were never checked by the "
            "critic (not the same as checked and failed)."
        )

    if dry_run:
        typer.echo("\n(dry run — nothing uploaded)")
        return

    try:
        token = oauth.get_access_token()
    except (oauth.MissingCredentials, RuntimeError) as e:
        typer.echo(f"\n{e}")
        raise typer.Exit(1) from e

    typer.echo("\nUploading...")
    try:
        result = youtube.upload(ep.video_path, metadata, token)
    except (RuntimeError, FileNotFoundError) as e:
        typer.echo(f"\nUpload failed: {e}")
        raise typer.Exit(1) from e

    typer.echo(f"\nUploaded: {result.url}")
    typer.echo(f"Privacy:  {result.privacy}")
    if result.privacy == "private":
        typer.echo("Promote it in YouTube Studio when you're happy with it.")
