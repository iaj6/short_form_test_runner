"""CLI entry point — typer-based command interface."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from shortform.config import PROJECT_ROOT, list_strategies, load_settings, load_strategy
from shortform.models.script import Script
from shortform.models.video import Video
from shortform.pipeline.context import PipelineContext
from shortform.pipeline.runner import PipelineRunner
from shortform.stages.assembly import AssemblyStage
from shortform.stages.script_gen import ScriptGenStage
from shortform.stages.tts import TTSStage
from shortform.stages.visual_gen import VisualGenStage
from shortform.store.db import Database
from shortform.visuals import get_backend, list_backends

app = typer.Typer(
    name="shortform",
    help="Automated short-form video content creation.",
    no_args_is_help=True,
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

    # Resolve visual backend
    backend_name = visual_backend or settings.visuals.backend
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
    stages = [ScriptGenStage(), TTSStage(), VisualGenStage(backend=backend), AssemblyStage()]
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
    typer.echo(f"  Segments: {ctx.script.segment_count} (~{ctx.script.total_duration:.1f}s estimated)")
    typer.echo("")
    for seg in ctx.script.segments:
        typer.echo(f"  [{seg.index}] ({seg.estimated_duration:.1f}s) {seg.narration}")
        if seg.visual_prompt:
            typer.echo(f"      visual: {seg.visual_prompt[:90]}{'...' if len(seg.visual_prompt) > 90 else ''}")
    typer.echo("")
    typer.echo(f"Next: uv run shortform generate-from-script '{out_path}'")


@app.command("generate-from-script")
def generate_from_script(
    script_path: Path = typer.Argument(..., help="Path to a script JSON file"),
    visual_backend: str | None = typer.Option(
        None, "--visual-backend", "-vb", help="Visual backend (pillow, veo)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run TTS → visuals → assembly from an existing script JSON.

    Skips script_gen entirely (no Claude call, no ANTHROPIC key needed for
    that stage). The script's strategy_name determines which strategy YAML
    is loaded for TTS/visuals/music config.
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

    backend_name = visual_backend or settings.visuals.backend
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

    stages = [ScriptGenStage(), TTSStage(), VisualGenStage(backend=backend), AssemblyStage()]
    runner = PipelineRunner(stages=stages, db=db)

    video = Video(
        strategy_name=script_obj.strategy_name,
        topic=script_obj.topic,
        title=script_obj.title,
        script_id=script_obj.id,
    )
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

    ctx = asyncio.run(runner.run(ctx, resume_from="script_gen"))

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
