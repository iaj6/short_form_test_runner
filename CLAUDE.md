# Project Context

Short-form video generation pipeline targeting YouTube Shorts and Instagram Reels. Intentionally identity-separated from the user's personal accounts (anonymous channels). Goal is a fully closed-loop autonomous system — no human review at steady state. Python, local-first, free/open tools where possible.

## Pipeline Architecture

Linear stages with SQLite checkpointing. Resume-from-stage supported.

```
ScriptGenStage  →  TTSStage  →  VisualGenStage  →  AssemblyStage  →  (PublishStage — stub)
   (Claude)       (Edge TTS)     (Veo / Pillow)      (FFmpeg)
```

Key files:
- `src/shortform/pipeline/runner.py` — orchestration, checkpointing
- `src/shortform/stages/` — one module per stage
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

1. **F5-TTS proof-of-concept** on the M2 Pro spare Mac — biggest unknown. If the cloned theatrical voice doesn't sell the concept, nothing else matters.
2. Character reference image (Imagen via Gemini API)
3. `gothic_vignette.yaml` strategy config
4. Veo backend: reference-image anchoring + seed control
5. Source gothic/melancholic royalty-free music (new category)
6. End-to-end test: one full ~30s video

## Machine Topology

User has two machines for this project:

- **Primary dev machine** (where the repo was created): iteration, code, Veo API calls (hit Google regardless). This is where the user typically works.
- **Spare M2 Pro MacBook** (where you probably are now): long-running local inference. F5-TTS runs here. Eventually Whisper (for speech-accurate subtitle timing) and scheduled overnight batch generation.

**Current architecture (simplest):** run the whole pipeline on whichever machine. Later, when F5-TTS integration stabilizes, we may split — main machine orchestrates, spare machine exposes F5-TTS as a FastAPI service on the LAN. For now, everything-local-to-current-machine is fine.

## Data NOT in the Repo

These paths are gitignored and need recreating or syncing:

- `.env` — API keys. Copy from primary machine or rebuild from `.env.example`. Needs `ANTHROPIC_API_KEY` and `GOOGLE_GEMINI_API_KEY`.
- `data/videos/` — generated output, ~200MB on primary. Don't sync, regenerate when needed.
- `data/assets/` — per-segment frames, ~500MB. Intermediate only, regenerates.
- `data/music/ambient/` and `data/music/upbeat/` — royalty-free tracks. Need to source/sync separately. New `data/music/gothic/` category coming for the Bartholomew work.
- `data/*.db` — SQLite state, machine-specific, don't sync.
- Voice reference audio (for F5-TTS) — will live at `data/voices/bartholomew_reference.wav` or similar, gitignored.

## Key Decisions Already Made (don't re-debate without cause)

- **Python, not TypeScript** — better ML/video ecosystem. User defaults to TS but agreed Python is the right pick here.
- **Local F5-TTS over ElevenLabs** — user doesn't want to pay-to-play with ElevenLabs free-tier character limits. F5-TTS is open, high-quality, runs fine on M2 Pro.
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
