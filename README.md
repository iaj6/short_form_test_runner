# shortform

A pluggable Python pipeline for fully-automated short-form video generation. Strategies defined in YAML describe the *kind* of video to produce; the pipeline runs script → variant selection → TTS → visuals → assembly and emits a finished vertical MP4.

Designed for unattended batch operation, with an editorial gate available when you want to review scripts before committing to the expensive generation stages.

## What it does

Given a strategy YAML (which encodes a content type — prompts, tone, voice, visual style, music) the pipeline:

1. **Generates a script** via Claude using the strategy's system + template prompts.
2. **Picks a hero-reference variant** per segment via Claude (for strategies that ship a variant library — see below).
3. **Synthesizes narration** via Edge TTS (free, fast) or F5-TTS (open-source voice cloning, runs locally).
4. **Generates per-segment visuals** via Pillow (gradient stills with Ken Burns) or Veo image-to-video (anchored to the variant). Long segments get multiple Veo clips chained via last-frame extraction so motion stays continuous across sub-clip cuts.
5. **Assembles** in ffmpeg with crossfade transitions, sidechain-compressed music ducking, animated subtitles (when the TTS backend provides word timings), and fast-start MP4 remuxing.

Output is 1080×1920 vertical, suited for YouTube Shorts / Instagram Reels.

## Quick start

Prereqs: Python ≥ 3.12, [`uv`](https://docs.astral.sh/uv/), `ffmpeg` in PATH.

```bash
git clone https://github.com/iaj6/short_form_test_runner.git
cd short_form_test_runner
uv sync

cp .env.example .env
# Edit .env — set at minimum ANTHROPIC_API_KEY (required) and
# GOOGLE_GEMINI_API_KEY (only if you'll use the Veo visual backend)

# Full closed-loop run on a built-in strategy
uv run shortform generate -s motivation_quotes
```

The first run with the default Pillow visual backend produces a video in under a minute and costs ~$0.01 in Claude tokens. Veo runs are slower (~20–30 min) and cost ~$3–5 per video at the current multi-clip cadence.

## Architecture

Linear pipeline stages with SQLite checkpointing and `resume_from` support:

```
ScriptGenStage  →  VariantSelectionStage  →  TTSStage          →  VisualGenStage             →  AssemblyStage
   (Claude)         (Claude — optional)        (Edge or F5-TTS)     (Pillow or Veo,                (FFmpeg)
                                                                     multi-clip + last-frame
                                                                     chaining for long segments)
```

Two CLI entry points:

```bash
# A) Closed loop — script through finished video in one command.
uv run shortform generate -s gothic_vignette --visual-backend veo

# B) Editorial gate — script first, review/edit, then continue.
uv run shortform script -s gothic_vignette -t "an old photograph found in a drawer"
# Writes data/scripts/<id>.json. Open it, tweak narration, save.
uv run shortform generate-from-script data/scripts/<id>.json --visual-backend veo
```

Strategy YAMLs are overlay-only — they override `config/default.yaml` selectively. The strategy specifies *what* kind of video to make; the pipeline knows *how* to make it.

## Strategies

A strategy is one file in `config/strategies/`. The repo ships three:

| Strategy | What it produces | Backends used |
|---|---|---|
| `motivation_quotes` | Calm reflective single-insight pieces with abstract cinematic visuals | Edge TTS + Pillow (or Veo) |
| `tech_tips` | Energetic "insider knowledge" tech tips with close-up device visuals | Edge TTS + Pillow (or Veo) |
| `gothic_vignette` | Deadpan claymation-style gothic vignettes about modern dread. Showcase strategy — uses F5-TTS voice cloning, Veo image-to-video, and the per-segment hero variant system. | F5-TTS + Veo (variant-anchored) |

Authoring a new strategy is just writing a YAML file. The schema is documented in `src/shortform/config.py` (`StrategyConfig`); the existing strategies are the best references.

A strategy file declares:
- **Content** — tone, style, target duration, segment count, default voice.
- **Prompts** — `system` (sets the writer persona, style rules, anti-patterns, optional few-shot examples) and `template` (the per-video script-generation request).
- **Topics** — a list the script-gen stage randomly samples from, unless `--topic` is passed on the CLI.
- **TTS config** (optional) — pick a backend, set voice/rate or, for F5-TTS, the reference audio path + transcript.
- **Visuals config** — backend-specific. For Veo: optional `reference_image` for character consistency or `variants_manifest` for per-segment scene matching.
- **Music config** — which directory in `data/music/` to pick background tracks from.

## Backends

### TTS

| Backend | Cost | Quality | Notes |
|---|---|---|---|
| **Edge TTS** | Free | Good for clear-narration content | Provides word-level timings for animated subtitles. Default. |
| **F5-TTS** | Free (local) | Excellent voice cloning from a 6–12s reference clip | Runs in its own venv outside the project (kept slim — torch is heavy). No word timings. |

Pick via `tts.backend` in the strategy YAML.

### Visuals

| Backend | Cost | Output | Notes |
|---|---|---|---|
| **Pillow** | Free | Stills with Ken Burns animation in assembly | Default. Good for cinematic-quote-over-abstract-visual content. |
| **Veo** (Gemini API) | ~$0.50/clip | 8-second image-to-video clips | Anchors to a reference image. Hard-locked at ~8s per clip — long narration uses multi-clip per segment. |

Pick via `--visual-backend` on the CLI, or set `visuals.backend` in `config/default.yaml`.

A `kling_backend.py` / local-model backend can be slotted in by mirroring `src/shortform/visuals/veo_backend.py` and registering it in `src/shortform/visuals/registry.py`. The architecture is intentionally backend-agnostic.

## Hero variant system

The showcase feature, and one of the more useful pieces of the pipeline.

**The problem.** A locked character reference image (say, a skeleton sitting at a desk) is great for visual identity — every video features the same character — but Veo struggles when the segment's content asks for a different scene. Ask Veo to animate "the skeleton at his desk" while the prompt describes "pressing buttons on a dishwasher" and you get a Frankenstein hybrid: a dishwasher shoved into a Victorian study, the character morphed to reach for an object that doesn't fit the scene, motion uncanny because the scene logic is incoherent. Worse, the morphing increases safety-filter rejection rates.

**The fix.** Per-segment hero variants. Generate a small library of scene-specific reference images of the same character — same face, same wardrobe, same proportions, different settings. Then a `VariantSelectionStage` between script gen and TTS asks Claude to pick the best-fitting variant for each segment based on its narration and visual prompt. Visual gen uses the per-segment variant as the base frame Veo animates.

**Generating a variant library.** Use Nano Banana Pro (Gemini 3 Pro Image) to *edit* a locked character image into new scenes — image editing preserves identity dramatically better than re-generating from scratch. The repo's `scripts/generate_bartholomew.py --edit-variants` shows the pattern; ~$0.04 per variant.

The gothic_vignette strategy ships a six-variant library (study/laptop/kitchen/grocer/phone/window) plus its `variants_manifest` config pointing at `data/character_refs/variants.yaml`. Strategies without a variants manifest skip the selection stage entirely — it's purely additive.

## Editorial workflow

The closed-loop `generate` command is the default mode. But the script is where ~80% of a video's quality lives, and Claude's first draft isn't always the script worth committing $3–5 of Veo credits to. The two-command split exists for that case:

```bash
# Generate 5 scripts on different topics, pay ~$0.05 in Claude tokens total
for topic in "rent increasing" "the dishwasher" "doom-scrolling" "Sunday evening" "the news"; do
  uv run shortform script -s gothic_vignette -t "$topic"
done

# Open the JSON files in data/scripts/, read them, keep the 1-2 you like
# Edit narration directly in the JSON if any line wants tweaking

# Generate visuals + audio only for the ones you picked
uv run shortform generate-from-script data/scripts/<best_one>.json --visual-backend veo
```

The script JSON format is documented by example — open one to see the shape. Manually-set `hero_variant` per segment is respected (the selection stage skips re-picking).

## Configuration

- `config/default.yaml` — base settings. LLM model, TTS defaults, video dimensions/bitrate/codec, music defaults, paths.
- `config/strategies/<name>.yaml` — per-strategy overlays. Only specify what you want to override.
- `.env` — API keys (`ANTHROPIC_API_KEY` required for any pipeline run; `GOOGLE_GEMINI_API_KEY` required for Veo and for Nano Banana Pro variant generation).

The config loader is `src/shortform/config.py`. Pydantic settings + YAML overlay; env vars use double-underscore nesting (e.g., `LLM__MODEL=claude-opus-4-1`) if you prefer overriding without editing files.

## Costs

Honest numbers, per video, at the current implementation:

| Backend mix | Wall time | Approx cost |
|---|---|---|
| Edge TTS + Pillow | < 1 min | ~$0.01 (Claude script gen) |
| Edge TTS + Veo | 15–25 min | ~$3–5 (Veo dominates) |
| F5-TTS + Pillow | ~8 min | ~$0.01 |
| F5-TTS + Veo (gothic_vignette, multi-clip + variants) | 20–30 min | ~$3.50–$5 |

Variant-library generation (one-time per character): ~$0.20 for 5 Nano Banana Pro edits.

## Production notes

The pipeline accumulated retry/robustness layers from real failures — they're worth knowing about before you remove anything:

- **Veo 5xx** (8s base backoff, 4 attempts) — for transient Gemini API outages.
- **Veo 429 rate-limit** (30s base backoff, 4 attempts) — Gemini's per-minute quota is reachable in batch runs.
- **Veo 429 credits-depleted** (fail-fast) — distinguishes "wait it out" from "your balance is gone."
- **Veo safety-filter retry** (1 retry before falling back to a Pillow still) — Veo's filter is statistical; same input often succeeds on retry.
- **F5-TTS subprocess retry** (2 attempts) — PyTorch MPS occasionally segfaults at model load on Apple Silicon.

And in assembly, per-input normalization before xfade chains: `settb=AVTB,setpts=PTS-STARTPTS,fps=N,scale=W:H,format=PIXFMT` for video; `asettb=AVTB,asetpts=PTS-STARTPTS,aresample=R` for audio. Veo's outputs aren't consistent on timebase or framerate across calls and xfade rejects mismatches.

## Limitations

- **No automated publishing yet.** The `PublishStage` is a stub; uploads to YouTube/Instagram are manual via the platforms' UIs. Wiring the YT Data API v3 OAuth flow is on the Phase 2 list.
- **Music selection is random** within the configured genre directory. A `tracks.yaml` manifest format with mood tags is in place; a selector that matches tracks to segment content per video is Phase 2.
- **F5-TTS doesn't emit word timings**, so the assembly stage's animated phrase-level subtitle path is dead-code for F5-TTS strategies. Post-hoc Whisper alignment would re-enable it.
- **Veo is expensive** at the current per-clip cadence. Tightening narration toward 12–14s per segment (instead of the typical 16–22s) would drop ~30% of Veo cost per video; this is a strategy-prompt-tuning question, not a code one.
- **Local open-source video models** (HunyuanVideo, Wan 2.2) were investigated for cost-free generation but require ≥48GB unified memory on Apple Silicon to avoid swap thrashing. Documented as a Phase 0 attempt that didn't pan out at 32GB.

## License

MIT. See `LICENSE`.
