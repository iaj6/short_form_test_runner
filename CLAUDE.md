# Project Context

`shortform` is a pluggable Python pipeline for short-form video generation (YouTube Shorts / Instagram Reels formats). A strategy YAML describes a content type; the pipeline produces a finished 1080×1920 MP4. Designed for unattended autonomous runs with an editorial-gate option (script-first workflow) for cases where the script deserves human review before committing the expensive generation stages.

The README is the public-facing entry point. This file is the orientation doc for anyone (human or LLM) working *on* the codebase — design rationale, gotchas baked in from real failures, and where the cliff edges are.

## Pipeline Architecture

Linear stages with SQLite checkpointing. `resume_from="<stage_name>"` skips up to and including the named stage; used by the `generate-from-script` CLI to pick up where `script` left off.

```
ScriptGenStage → VariantSelectionStage → TTSStage         → VisualGenStage          → AssemblyStage
   (Claude)        (Claude — optional)     (Edge / F5-TTS)    (Pillow / Veo,             (FFmpeg)
                                                                multi-clip + chained)
```

Two CLI entry points:
- `shortform generate -s <strategy>` — full closed-loop run.
- `shortform script -s <strategy> [-t <topic>]` followed by `shortform generate-from-script <path>` — editorial gate. Script JSON written to `data/scripts/<id>.json` for review; the second command picks up via `resume_from="script_gen"`.

Key files:
- `src/shortform/pipeline/runner.py` — orchestration, checkpointing, `resume_from` skip semantics.
- `src/shortform/stages/` — one module per stage.
- `src/shortform/stages/variant_select.py` — picks per-segment hero variant from a manifest via Claude tool-use; no-op when the strategy doesn't declare `visuals.variants_manifest`.
- `src/shortform/tts/` — pluggable TTS backends (Edge, F5-TTS); strategy picks via `tts.backend`.
- `src/shortform/visuals/` — pluggable visual backends (Pillow, Veo). Veo defaults to `veo-3.1-generate-preview`.
- `src/shortform/stages/assembly.py` — FFmpeg-heavy. Per-input timebase+framerate normalization before xfade, multi-clip concat per segment, sidechain music ducking, Ken Burns on stills, animated subtitles.
- `config/strategies/*.yaml` — strategy overlays.
- `config/default.yaml` — base settings.

## Strategy system

A strategy is one YAML file in `config/strategies/`. It overlays `config/default.yaml` selectively. Schema is `StrategyConfig` in `src/shortform/config.py`:

- `content` — tone, style, target duration, segment count, default voice.
- `prompts.system` + `prompts.template` — drive script generation. Few-shot examples go here.
- `topics` — random sample pool unless `--topic` is passed.
- `tts.backend` + backend-specific params — `edge` (voice/rate/volume) or `f5_tts` (ref_audio, ref_text, model, speed, cfg_strength).
- `visuals` — backend-tunable params + `variants_manifest` if the strategy ships a hero-variant library.
- `music` — directory under `data/music/` to randomly pick background tracks from.

Strategies in the repo:

| Strategy | Backends | Showcase of |
|---|---|---|
| `motivation_quotes` | Edge TTS + Pillow (or Veo) | Minimal example — basic strategy without variants or voice cloning |
| `tech_tips` | Edge TTS + Pillow (or Veo) | Same shape as motivation_quotes, different content slot |
| `gothic_vignette` | F5-TTS + Veo with variant manifest | **Canonical example** — uses every advanced feature. Bartholomew clay-skeleton character running through Burton/Snicket-style vignettes about modern dread. Asset library + variants live in `data/character_refs/`. |

When iterating on the pipeline itself, `gothic_vignette` is the integration-test strategy because it exercises every code path. New strategies authored downstream don't need any of this; minimal strategies look like `motivation_quotes`.

## Hero variant system

The non-obvious architectural piece worth understanding before modifying anything in visual_gen or variant_select.

**Problem.** A single locked reference image works for visual identity (same character every video) but fails when the segment's content asks for a different scene than the reference shows. Veo will Frankenstein the two together — original-scene environment morphed to fit unrelated-content prompts — and the morphing tends to (a) look uncanny and (b) trip safety filters more often than the clean reference would.

**Solution.** A library of scene-variant images of the same character: same face, same wardrobe, same proportions, different settings. `VariantSelectionStage` runs between `script_gen` and `tts`, asking Claude (tool-use) to pick the best variant per segment based on narration + visual_prompt. `VisualGenStage` resolves the per-segment `hero_variant` key via `variants_manifest` and passes the resolved path as Veo's base frame.

**Generating variants.** Use Nano Banana Pro (Gemini 3 Pro Image) to *edit* the locked character image into new scenes. Image editing preserves identity dramatically better than text-to-image regeneration. `scripts/generate_bartholomew.py --edit-variants` shows the pattern; the manifest at `data/character_refs/variants.yaml` is the source of truth for what's available.

**For strategies without variants.** The resolver falls back to the strategy's singular `reference_image` config (legacy single-anchor behavior), so other strategies that don't ship a manifest aren't affected.

## Multi-clip + last-frame chaining

Veo is hard-locked at ~8 seconds per clip; F5-TTS typically narrates 14–22 seconds per segment. The gap is bridged in `VisualGenStage`:

1. Compute `n_clips = ceil(audio_duration / CLIP_TARGET_SECONDS)` per segment (7.5s target accounts for inter-clip xfade overlap).
2. Generate clip 0 anchored to the segment's hero variant.
3. For each subsequent clip, extract the last frame of the previous clip (`ffmpeg -sseof -0.1 ... -frames:v 1`) and pass it as Veo's starting frame (`chain_from` config key → resolved as the `image` input). The chained clips continue motion from where the previous clip ended.
4. `AssemblyStage` concats the sub-clips with a small video-only xfade before muxing with the segment's audio.

Tradeoff: clips 2+ within a segment lose the hero-anchor since they chain from the previous clip's last frame. In practice this is fine because Veo's image-to-video preserves character/world reliably within an 8s window, and segments rarely need more than 3 chained clips. Each segment re-anchors to the hero (or hero variant) at clip 0, so drift can't compound across segments.

## Retry layers (each one was added after a real failure — don't speculatively remove)

- **Veo 5xx** (`_submit_with_retry`, veo_backend.py) — 4 attempts, 8s base exponential backoff. Added after a 503 killed an end-to-end run.
- **Veo 429 rate-limit** (same path, longer backoff) — 30s base, 4 attempts. Added after batch runs hit the per-minute Gemini quota.
- **Veo 429 credits-depleted (fail-fast variant)** — sniffs the error message for "credits"/"depleted"/"billing" and skips retry. Added after a 30s × 4 backoff wasted 7.5 minutes retrying a non-retryable balance issue.
- **Veo safety-filter rejection retry** (in `VeoBackend.generate`) — 2 attempts. Veo's safety filter is statistical; same input often succeeds on retry. Added after gothic-vignette runs had multiple segments fall back to Pillow stills.
- **F5-TTS subprocess retry** (`SUBPROCESS_MAX_ATTEMPTS = 2` in f5_backend.py) — handles SIGSEGV-at-MPS-load (exit -11), a known PyTorch-on-Apple-Silicon transient.

ffmpeg gotchas baked into `assembly.py`: per-input normalization before xfade chains via `settb=AVTB,setpts=PTS-STARTPTS,fps=N,scale=W:H,format=PIXFMT` (video) and `asettb=AVTB,asetpts=PTS-STARTPTS,aresample=R` (audio). Veo's outputs vary in timebase (1/12288 vs 1/15360 seen) and framerate (24 vs 25 fps seen) across calls; both trip xfade without normalization.

## F5-TTS setup notes

- Isolated venv at `~/.venvs/f5-tts` (kept separate from project venv to avoid torch + model weights bloating the slim shortform deps).
  ```
  uv venv ~/.venvs/f5-tts --python 3.12
  uv pip install --python ~/.venvs/f5-tts/bin/python f5-tts
  ```
  Requires `ffmpeg` (Homebrew).
- Reference audio + transcript live under `data/voices/` (gitignored). Strategies that use F5-TTS reference these in their `tts.ref_audio` / `tts.ref_text` fields.
- **F5-TTS silently clips `--ref_audio` to ~12 seconds.** It logs `Audio is over 12s, clipping short.` and proceeds, but uses the full `--ref_text` for rate estimation — a transcript longer than the clipped audio produces rushed output (e.g., 36 words generated as 3.3s instead of 17s). Always trim references to 8–12s with a matching partial transcript. `ffmpeg -af "silencedetect=noise=-30dB:d=0.4"` is the easiest way to find sentence-boundary cut points.
- Pipeline integration in `src/shortform/tts/f5_backend.py` subprocesses the CLI per segment. Each invocation pays the model-load cost (~30s–3min depending on cache state). For production batch runs a persistent service would amortize this; deferred.

## Local open-source video models

Attempted (HunyuanVideo I2V on a 32GB Apple Silicon machine) and shelved. The model + text encoders + activations exceeded available unified memory and started disk-swap thrashing. Smaller open models (LTX-Video, Wan 2.2 quantized) or a Mac Studio with ≥48GB would be the path forward if this revisits. Not pursued urgently because the variant system improved Veo's reliability enough that the credits-vs-quality tradeoff is more tolerable.

## Data NOT in the repo (.gitignored, recreate locally)

- `.env` — API keys. `ANTHROPIC_API_KEY` is required; `GOOGLE_GEMINI_API_KEY` is required for Veo and for variant generation.
- `data/videos/` — generated output.
- `data/assets/` — per-segment frames + intermediate audio. Regenerates.
- `data/music/<category>/` — royalty-free tracks (large, licensed per-source).
- `data/scripts/` — script JSONs from the editorial workflow.
- `data/voices/` — F5-TTS reference audio + per-machine test outputs.
- `data/*.db` — SQLite pipeline state (machine-local).
- `~/.venvs/f5-tts/` — separate Python 3.12 venv with f5-tts installed (lives outside the repo by design).

## Phase 2 wishlist

These are real next-step ideas, not commitments. None block current functionality.

- **Music selector** — `data/music/<category>/tracks.yaml` has mood-tag schema in place; a selector matching tracks to vignette content per video is the cheapest remaining quality win.
- **Publish automation** — `src/shortform/stages/publish.py` is a stub. YT Data API v3 OAuth flow + resumable upload is ~5–6 hours of work. Deferred until a published channel has data to validate against.
- **Cost optimization for Veo strategies** — narration tightening from 16–22s/segment toward 12–14s/segment drops ~30% of Veo cost per video. Strategy-prompt-tuning, not code.
- **Whisper subtitle alignment** — F5-TTS doesn't emit word timings; the existing animated-subtitle path in assembly is dead-code for F5-TTS strategies. Post-hoc Whisper alignment would re-enable it.
- **Alternate visual backends** — Kling for looser-filter content, or another local-model attempt on heavier hardware. The `src/shortform/visuals/registry.py` pattern makes adding one straightforward.

## Key decisions already made (don't re-debate without new info)

- **Python, not TypeScript** — better ML/video ecosystem.
- **Local F5-TTS over hosted ElevenLabs** — free, high quality, no character limits.
- **Veo image-to-video, not text-to-video** — stronger consistency, the reference-image anchor extends naturally to the variant system.
- **Linear stages with SQLite checkpointing** — solid; improvements happen at the *content* layer, not the orchestration layer.
- **Claude for script generation, not Gemini** — quality difference matters for tone-sensitive writing.
- **No publishing automation yet** — Phase 2.
- **One repo, multiple strategies** — keeping motivation_quotes and tech_tips alongside gothic_vignette makes the pipeline's flexibility legible.
