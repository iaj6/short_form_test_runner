# Project Context

Short-form video generation pipeline targeting YouTube Shorts and Instagram Reels. Intentionally identity-separated from the user's personal accounts (anonymous channels). Goal is a fully closed-loop autonomous system — no human review at steady state. Python, local-first, free/open tools where possible.

## Pipeline Architecture

Linear stages with SQLite checkpointing. Resume-from-stage supported.

```
ScriptGenStage → VariantSelectionStage → TTSStage         → VisualGenStage    → AssemblyStage  → (PublishStage — stub)
   (Claude)        (Claude — per-segment    (Edge / F5-TTS)    (Veo / Pillow,      (FFmpeg)
                    hero variants)                              multi-clip + chained)
```

Two CLI entry points to this pipeline:
- `shortform generate -s <strategy>` — full closed-loop run (script → TTS → visuals → assembly).
- `shortform script -s <strategy>` followed by `shortform generate-from-script <path>` — editorial gate. Script JSON is written to `data/scripts/<id>.json` for review/edit; the second command picks up from where the script left off via `resume_from="script_gen"`.

Key files:
- `src/shortform/pipeline/runner.py` — orchestration, checkpointing, `resume_from` skip semantics
- `src/shortform/stages/` — one module per stage
- `src/shortform/stages/variant_select.py` — picks per-segment hero variant from a manifest via Claude tool-use; no-op when the strategy doesn't declare `visuals.variants_manifest`
- `src/shortform/tts/` — pluggable TTS backends (Edge, F5-TTS); strategy picks via `tts.backend`
- `src/shortform/visuals/` — pluggable backends (Pillow, Veo; Grok stub); Veo is on `veo-3.1-generate-preview` by default
- `src/shortform/stages/assembly.py` — the heavyweight: FFmpeg with timebase+framerate+format normalization before xfade, multi-clip concat per segment, sidechain music ducking, Ken Burns on stills, animated subtitles
- `config/strategies/*.yaml` — strategy-specific prompts, voices, visual style, variants manifest pointer, music category
- `config/default.yaml` — base settings
- `data/character_refs/variants.yaml` — manifest of 6 Bartholomew hero variants (study/laptop/kitchen/grocer/phone/window); the source-of-truth list that VariantSelectionStage picks from

## Current Strengths

The assembly stage is genuinely sophisticated — don't rewrite it. It does:
- Phrase-level animated subtitles (PNG overlays with enable-timing)
- Sidechain-compressed background music ducking under narration
- Ken Burns (zoompan) on still images
- Frame-accurate xfade crossfades between clips
- Fast-start MP4 remuxing

## Current Weaknesses (motivating the active initiative)

Output reads as "AI short-form" because:
- Every video structurally identical (3 segments, hook→reveal→payoff)
- Visual monotony: single 8s Veo clip per segment, one zoom style, static gradient base frames
- Edge TTS voices sound like news anchors, not characters
- No recurring visual identity / character / world
- Prompts describe nouns, not cinematography
- Music bed is 2-5 tracks looped across everything

## Active Initiative: Bartholomew — Gothic Vignette Format

**Creative brief.** A recurring clay-skeleton protagonist ("Bartholomew") in a claymation gothic world. Tone is Lemony Snicket meets Tim Burton meets Edward Gorey — dry, deadpan, darkly comic meditations on *modern* dread: layoffs, AI replacement, inflation, doom-scrolling, rent increases, dating apps, the news. Format reference: Burton's *The Melancholy Death of Oyster Boy* — tiny tragicomic vignettes where the narrator takes an amused, mildly-menacing view of suffering.

**Why this pick.** Stop-motion's inherent jank masks AI artifacts. Gothic deadpan is unclaimed territory in AI short-form (everyone else is chasing optimism or rage). Single recurring protagonist is more tractable for consistency than multi-character. Subject matter is evergreen — no trend-chasing required.

**Character details (locked).**
- Clay skeleton, slightly too-formal outdated wardrobe (moth-eaten tweed cardigan, oversized bowtie, crooked bowler hat, oversized round spectacles)
- Setting: dim Victorian study that anachronistically contains a laptop, modem, unpaid bills next to an urn
- Physical comedy hooks: jaw clacks when resigned, fingers rattle when typing, eye-socket shadows substitute for blinking
- Name: **Bartholomew** (or "Bart" in his own head)

**Narrator voice.** Price-*adjacent* (theatrical gothic, arched vowels, operatic phrasing). Not cloning Vincent Price directly — estate is litigious about AI voice clones. Plan is for the user to record a 30-60s reference clip of their own theatrical narrator voice, then clone via F5-TTS.

**Technical approach for visual consistency.**
1. Generate one hero reference image of Bartholomew via Gemini Imagen. Iterate on 6-8 variants, lock one.
2. Extend Veo backend to accept a `reference_image` from strategy config, use it as the base frame for *every* segment (not the current per-segment Pillow gradient). This is the primary consistency signal — stronger than Veo seeds alone.
3. Also pass a fixed seed per video for belt-and-suspenders consistency.
4. Strategy YAML (`gothic_vignette.yaml`) with heavy Burton/claymation style descriptors, character description, script template tuned for the deadpan-vignette format.

## Build Order (de-risk hardest thing first)

1. ~~F5-TTS proof-of-concept + pipeline integration~~ **DONE** (2026-05-12). Voice clone sells the gothic-deadpan concept; pipeline subprocesses `~/.venvs/f5-tts/bin/f5-tts_infer-cli` per segment. F5-TTS has a SIGSEGV-at-MPS-load transient on Apple Silicon (exit -11) — retry is now built in (`SUBPROCESS_MAX_ATTEMPTS = 2`). See "F5-TTS Setup Notes" below.
2. ~~Character reference image + variant library~~ **DONE**. Locked hero at `data/character_refs/bartholomew_hero.png` (Imagen 4 generation, `scripts/generate_bartholomew.py`). On 2026-05-14 we added 5 scene-variant siblings via Nano Banana Pro image-editing (study/laptop/kitchen/grocer/phone/window) — the same character preserved exactly (skull-crack, bowtie, wardrobe), different scenes. Generation via `scripts/generate_bartholomew.py --edit-variants`.
3. ~~`gothic_vignette.yaml` strategy config~~ **DONE**. Has the system prompt with six canonical Bartholomew vignettes as few-shot examples, 26 modern-dread topics, `tts.backend: f5_tts` with ref_audio/ref_text params, and `visuals.variants_manifest: data/character_refs/variants.yaml` for per-segment hero selection.
4. ~~Veo backend~~ **DONE** with substantial robustness. Reads `reference_image` from per-segment config (set by VisualGenStage's variant resolver). On `veo-3.1-generate-preview`. Has retry layers for: 5xx transient errors (8s base backoff, 4 attempts), 429 rate-limit (30s base backoff, 4 attempts), safety-filter rejections (1 retry before falling back to Pillow). Fails fast on 429 when the message indicates depleted credits rather than rate-limit. `veo_seed` is no-op via Gemini API (would need Vertex AI).
5. ~~Multi-clip per segment + last-frame chaining + scene-coherence via variants~~ **DONE** (2026-05-13 to 14). Veo is hard-locked at ~8s per clip; F5-TTS narrates 12-22s per segment; the gap is bridged by generating ceil(audio_duration / 7.5) Veo clips per segment, chained via last-frame extraction so motion is continuous across sub-clip boundaries. Variant system added on 2026-05-14: each segment uses a scene-matched hero variant instead of always morphing the locked study scene. Validated to eliminate the safety-filter rejections that "skeletal hands in a kitchen" prompts were triggering.
6. ~~Royalty-free music sourcing~~ **DONE** locally. `data/music/gothic/tracks.yaml` + 6 downloaded MP3s. Selector logic to match tracks per-vignette by mood is not yet built; `assembly.py` still picks randomly. Tracked as Phase 2 work below.
7. ~~End-to-end test~~ **DONE**. 7+ Bartholomew videos generated across the two days; one ("The Colleague") manually uploaded to the channel.
8. ~~First YouTube Short published~~ **DONE** (2026-05-13). Channel: https://www.youtube.com/channel/UCH0aAvB6yTWic68j2Q0gsvg. Manual upload via youtube.com/upload while logged into the channel's Google account; automation deferred (see Phase 2).

**Phase 2 (next, when motivated):**
- **Music selector** — read `tracks.yaml` mood tags + segment narration, match track per video. Cheap quality win.
- **Publish automation** — YT Data API v3 OAuth flow + `src/shortform/stages/publish.py` implementation. ~5-6 hours; deferred until the channel proves it has any organic reach.
- **Cost optimization** — narration is currently 16-22s per segment which forces 2-3 Veo clips; pushing toward 12-14s segments would drop ~30% of Veo cost per video.
- **Whisper subtitle alignment** — F5-TTS doesn't emit word timings; the existing animated-subtitle path in assembly is dead-code for the gothic strategy. Post-hoc Whisper alignment would re-enable it.
- **Alternate visual backends** — Kling for safety-filter-tight content (looser filters than Veo), or local open-source models for cost-free generation (Phase 0 attempt of HunyuanVideo I2V on M2 Pro thrashed at 32GB — see `docs/m2_pro_phase_0_handoff.md`).

## Machine Topology

**Single-machine setup as of 2026-05-12.** Everything runs on the user's MacBook Air (Apple M4, 32GB RAM, macOS 15.6). The M4 + 32GB is plenty for F5-TTS (MPS), ffmpeg (hw H.264/H.265 encoders), and the rest of the pipeline at iteration scale. There is also a spare M2 Pro MacBook available for future batch jobs but it's not part of the current critical path — the original "primary = code, spare = inference" split assumed primary was the weaker box, which isn't true here.

The earlier plan to run F5-TTS as a FastAPI LAN service from the M2 Pro is **shelved** unless something forces it back (e.g., long overnight batch jobs where sustained throughput on the actively-cooled Pro beats the fanless Air). Don't pre-build it.

## F5-TTS Setup Notes

- Isolated venv at `~/.venvs/f5-tts` (kept separate from project venv to avoid pulling torch into the slim shortform deps). Install: `uv venv ~/.venvs/f5-tts --python 3.12 && uv pip install --python ~/.venvs/f5-tts/bin/python f5-tts`. Requires `ffmpeg` (brew).
- Reference voice for Bartholomew lives at `data/voices/bartholomew_reference.m4a` (54s original, gitignored) with corresponding `bartholomew_reference.txt`. The trimmed-for-F5-TTS clip is `data/voices/bartholomew_reference_trimmed.wav` (9.7s) — this is the path `gothic_vignette.yaml` actually feeds to F5-TTS. The original 54s recording is the canonical artifact — back it up off-machine; re-recording will not match.
- **F5-TTS clips `--ref_audio` to ~12 seconds.** It silently logs `Audio is over 12s, clipping short.` then proceeds. The full `--ref_text` is still used to estimate speaking rate, so a transcript longer than the clipped audio produces rushed output (e.g., 36 words generated as 3.3s instead of 17s). Always trim references to 8–12s with a matching partial transcript. Use `ffmpeg -af "silencedetect=noise=-30dB:d=0.4"` to find sentence-boundary cut points.
- MPS works on both the M2 Pro (where the PoC was first validated, ~63s first inference for ~17s output) and the M4 Air with 32GB (where the pipeline integration was validated, ~3min cold for the first segment due to model load + JIT; subsequent same-process inferences are fast but each subprocess invocation pays the load cost again).
- Pipeline integration in `src/shortform/tts/f5_backend.py` (subprocess the CLI per segment). Strategy YAMLs opt in via a `tts:` block — see `config/strategies/gothic_vignette.yaml` for the canonical example.

## Data NOT in the Repo

These paths are gitignored and need recreating or syncing:

- `.env` — API keys. Needs `ANTHROPIC_API_KEY` and `GOOGLE_GEMINI_API_KEY`. Rebuild from `.env.example` if missing.
- `data/videos/` — generated output. Don't sync, regenerate when needed.
- `data/assets/` — per-segment frames + intermediate audio. Regenerates.
- `data/music/{ambient,upbeat,gothic}/` — royalty-free tracks. `gothic/tracks.yaml` is committed and has URLs for re-downloading.
- `data/*.db` — SQLite state, machine-specific.
- `data/voices/` — F5-TTS reference audio + per-machine test outputs. Current locked reference: `data/voices/bartholomew_reference_trimmed.wav` (9.7s) with matching `bartholomew_reference.txt` (the ref_text). Also `data/voices/bartholomew_reference.m4a` (the full 54s original).
- `~/.venvs/f5-tts/` — separate uv-managed Python 3.12 venv with `f5-tts` installed from PyPI. Set up with: `uv venv ~/.venvs/f5-tts --python 3.12 && uv pip install --python ~/.venvs/f5-tts/bin/python f5-tts`. Lives outside the repo by design (heavy ML stack).

## Retry Layers (each one was added after a real failure — don't speculatively remove)

The pipeline accumulated four retry layers over the active development period. Each handles a different class of transient:

- **Veo 5xx (`_submit_with_retry` in veo_backend.py)** — 4 attempts, 8s base exponential backoff. Added after a 503 killed run 2 of the first end-to-end pipeline.
- **Veo 429 rate-limit (same retry path, longer backoff)** — 30s base, 4 attempts. Added after a 3-video batch hit the per-minute Gemini quota.
- **Veo 429 credits-depleted (fail-fast variant)** — sniffs the error message for "credits"/"depleted"/"billing" and skips retry. Added after the 30s × 4 backoff wasted 7.5 minutes retrying a non-retryable balance issue.
- **Veo safety-filter rejection (retry in `VeoBackend.generate`)** — 2 attempts. Veo's safety filter is statistical; same prompt + ref image often succeeds on retry. Added after "The Dishwasher" had 2/3 segments fall back to Pillow gradients due to filter trips.
- **F5-TTS subprocess (`SUBPROCESS_MAX_ATTEMPTS` in f5_backend.py)** — 2 attempts. F5-TTS occasionally segfaults at MPS model load on Apple Silicon (exit -11). Same input usually succeeds on retry.

ffmpeg gotchas baked into `assembly.py`: per-input normalization before xfade chains via `settb=AVTB,setpts=PTS-STARTPTS,fps=N,scale=W:H,format=PIXFMT` (video) and `asettb=AVTB,asetpts=PTS-STARTPTS,aresample=R` (audio). Veo's outputs vary in timebase (1/12288 vs 1/15360 seen) and framerate (24 vs 25 fps seen) across calls; both trip xfade without normalization.

## Key Decisions Already Made (don't re-debate without cause)

- **Python, not TypeScript** — better ML/video ecosystem. User defaults to TS but agreed Python is the right pick here.
- **Local F5-TTS over ElevenLabs** — user doesn't want to pay-to-play with ElevenLabs free-tier character limits. F5-TTS is open, high-quality, runs fine on the M4 Air.
- **Veo image-to-video, not text-to-video** — already gives stronger consistency. Reference-image anchoring extends this.
- **Keep the existing pipeline architecture** — linear stages + SQLite checkpointing + strategy YAML overlays are solid. The improvements are at the *content* layer, not the infra layer.
- **Claude (Anthropic) for script generation, not Gemini** — quality difference matters for tone-sensitive writing like this.
- **No publishing automation yet** — Phase 2. Focus on getting output quality to "actually watchable" first.

## Session-Specific Notes

- The user has a port registry at `~/.ports.json` — check it before binding any local server (e.g., F5-TTS FastAPI server).
- Strategy YAMLs are overlay-only — they override `default.yaml` selectively. See `src/shortform/config.py` for loading logic.
- `data/videos/` output naming: `{short_uuid}_{title}.mp4`. 13 assembled, 7 failed in the existing DB on primary machine.
- User is a pragmatic incrementalist — prefers working MVPs over elaborate designs. Ship in small testable chunks.

## Repo

`https://github.com/iaj6/short_form_test_runner` (private)
