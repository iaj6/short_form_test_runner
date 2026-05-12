# Project Context

Short-form video generation pipeline targeting YouTube Shorts and Instagram Reels. Intentionally identity-separated from the user's personal accounts (anonymous channels). Goal is a fully closed-loop autonomous system — no human review at steady state. Python, local-first, free/open tools where possible.

## Pipeline Architecture

Linear stages with SQLite checkpointing. Resume-from-stage supported.

```
ScriptGenStage  →  TTSStage         →  VisualGenStage  →  AssemblyStage  →  (PublishStage — stub)
   (Claude)       (Edge / F5-TTS)     (Veo / Pillow)      (FFmpeg)
```

Key files:
- `src/shortform/pipeline/runner.py` — orchestration, checkpointing
- `src/shortform/stages/` — one module per stage
- `src/shortform/tts/` — pluggable TTS backends (Edge, F5-TTS); strategy picks via `tts.backend`
- `src/shortform/visuals/` — pluggable backends (Pillow, Veo; Grok stub)
- `src/shortform/stages/assembly.py` — the heavyweight: FFmpeg, animated subtitles, sidechain music ducking, Ken Burns on stills, crossfade transitions
- `config/strategies/*.yaml` — strategy-specific prompts, voices, visual style, music category
- `config/default.yaml` — base settings

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

1. ~~F5-TTS proof-of-concept~~ ✓ **DONE** (2026-05-12) — voice clone sells the gothic-deadpan concept; greenlit. Pipeline integration also landed: `src/shortform/tts/f5_backend.py` subprocesses `~/.venvs/f5-tts/bin/f5-tts_infer-cli` per segment; `gothic_vignette.yaml` opts in via its `tts:` block. First-segment cold start is ~3min on M4 (model load + JIT). See "F5-TTS Setup Notes" below for install + the 12s ref-audio gotcha.
2. ~~Character reference image (Imagen via Gemini API)~~ **DONE** — locked at `data/character_refs/bartholomew_hero.png`. Generation script lives at `scripts/generate_bartholomew.py` (run with variants `--variant no_hat|wider|closer|standing` for alternate framing). Candidates pool is gitignored.
3. ~~`gothic_vignette.yaml` strategy config~~ **DONE** — `config/strategies/gothic_vignette.yaml` has system prompt with the six canonical Bartholomew vignettes baked in as few-shot examples, 26 modern-dread topics, Veo reference/seed fields wired, and now a `tts:` block selecting `f5_tts` with the ref_audio/ref_text/model params.
4. ~~Veo backend: reference-image anchoring + seed control~~ **DONE** — `src/shortform/visuals/veo_backend.py` reads `reference_image`, `veo_seed`, `veo_negative_prompt` from strategy config. Falls back to Pillow gradient gracefully when reference file missing. Tests in `tests/test_visuals.py`.
5. ~~Source gothic/melancholic royalty-free music~~ **DONE on the Air** — `data/music/gothic/tracks.yaml` lists 6 tracks with URLs/licenses/attribution. All 6 audio files downloaded locally. Future selector logic can read the manifest to match tracks to vignettes by mood/tempo/intensity and auto-include attribution lines in publish descriptions (current `assembly.py` still picks randomly from all audio in the directory — manifest is advisory until selector lands).
6. **End-to-end test:** one full ~30s video — now unblocked. Next concrete step.

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
