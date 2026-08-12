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

## Multi-speaker dialogue (`Segment.turns` + `VoiceCast`)

Built for adapted stage plays, where a segment is an exchange between characters rather than one narrator's monologue. Single-narrator strategies are completely unaffected — they declare no `tts.voices` and take the original code path.

**The data model.** `Segment.turns: list[Turn]`, where `Turn` is `{speaker, line, stage_direction}`. The critical design choice: **`narration` stays authoritative-looking and is *derived* from the turns** in `Segment.__post_init__` (joined `line`s, stage directions excluded — they're performance notes, never spoken). That means every existing `.narration` consumer — caption alignment, variant selection, Veo prompt building, the CLI script preview — keeps working without knowing that speakers exist. An explicitly-supplied `narration` wins over the derived join, so an adaptation can hand-write a cleaner flat text.

`Segment.from_dict` / `to_dict` centralize rehydration because a bare `Segment(**data)` leaves nested turns as raw dicts — which then explodes somewhere much later than the load. Both `Script.load_json` and `db.get_script` go through them. `narration` may be omitted from a dialogue segment's JSON; a segment with *neither* narration nor turns is a hard error rather than a silently silent segment.

**Voice resolution** (`src/shortform/tts/cast.py`). Cloning a reference voice doesn't scale past a handful of characters, so `VoiceCast` supports **mixing backends within one video**: F5-TTS clones for principals, Edge TTS for bit parts and crowds. Each speaker entry inherits the base/default config and overrides selectively, so a clone only names its own `ref_audio`. `backend` inside an entry switches that speaker's backend entirely.

```yaml
tts:
  backend: "f5_tts"          # default voice (also the narrator)
  ref_audio: "data/voices/narrator.wav"
  ref_text: "..."
  turn_gap: 0.28
  stage_direction_gap: 0.55
  voices:
    pere_ubu: { ref_audio: "data/voices/ubu.wav", ref_text: "...", speed: 1.05 }
    bordure:  { backend: "edge", voice: "en-GB-RyanNeural", rate: "+5%" }
```

Speaker keys are normalized (`normalize_speaker`), so an adaptation can emit `PÈRE UBU` straight from the source text and it resolves to `pere_ubu` — no mapping table. **An uncast speaker is fatal, not a fallback to the default voice**: a typo'd key would otherwise narrate an entire character in the wrong voice and you'd only find out by watching the finished video. `TTSStage.validate` resolves the whole cast up front so that failure lands before any inference time is spent.

**Gap policy.** `turn_gap` between turns; `stage_direction_gap` when the *next* line carries a stage direction (the beat belongs before the annotated delivery, not after it). The final turn gets no trailing silence — assembly controls inter-segment spacing and a trailing pad desyncs the video mux.

**Concatenation** (`src/shortform/tts/concat.py`). Mixed backends mean heterogeneous inputs (F5 emits 24kHz mono, Edge its own MP3 profile), which rules out ffmpeg's concat *demuxer* — it requires identical codec parameters and mangles mismatched ones. Uses the concat *filter* with per-input `aresample`/`aformat` normalization, the same lesson `assembly.py` already encodes for video xfade chains. Output matches `f5_backend._wav_to_mp3`'s profile, so **a multi-voice segment is indistinguishable from a single-voice one downstream** — assembly reads `seg.audio_path` and nothing else, which is why none of its 945 lines needed to change.

**Caption timings** are merged from per-turn timings with each turn's offset applied, but **all-or-nothing**: if any turn's backend emitted none (F5-TTS never does), the merge returns empty and the caller falls back to Whisper over the whole joined file. Partial timings would caption only the Edge-voiced lines and silently drop the cloned ones — worse than no captions, because it looks like it worked.

## Batch runner (`pipeline/batch.py`, `shortform batch`)

Renders many episodes unattended and reports what needs a human.

```
uv run shortform batch data/scripts/<series>e*.json -vb veo --report run.json
uv run shortform batch data/scripts/*.json --dry-run     # plan, spend nothing
```

**Deterministic video ids are the load-bearing piece.** `generate-from-script` mints a random id per run, so assets land in a fresh directory and a re-run regenerates everything. In batch mode **the video id IS the script id**, so assets are addressable (`data/assets/<script_id>/`), outputs sort by episode, and a re-run automatically reuses every clip the previous attempt paid for. Without this, resuming a half-finished batch costs full price twice.

**Skip / resume.** Finished episodes are skipped outright; partial ones resume from their existing clips via the reuse path in `visual_gen` — which is itself backend-gated, see below. Verified: a full 3-episode re-run took 1.2s and spent nothing.

**Failure isolation, with one exception.** An episode failing isolates to itself and the batch continues. Depleted Veo credits abort the run — every remaining episode would fail identically, so continuing just burns wall-clock. The credit sniff is shared with `veo_backend`'s retry ladder (`is_credits_error`) so the two can't drift.

**The report leads with what needs a human** — flagged clips and unverified clips, tracked separately because "failed continuity" and "never checked" are different problems. A flagged episode still counts as `ok`: it rendered, someone just has to look at it. `--report` writes JSON for scripting.

**The skip check is backend-aware**, via a `<script_id>.manifest.json` sidecar written next to each finished video recording the backend that produced it. Four cases, and the default in every ambiguous one is to RENDER — a wasted re-render costs money, but a wrong skip produces nothing and isn't noticed until someone goes looking for the file:

| state | action |
|---|---|
| manifest + video + backend matches | skip |
| manifest + video + different backend | render, warning that the previous output is being replaced |
| video but no manifest (e.g. from `generate-from-script`) | render — backend unknown, so we can't claim it's done |
| manifest but video deleted, or corrupt manifest | render |

Without this, a cheap `-vb pillow` test run marks every episode finished and a later Veo pass renders nothing at all. The manifest also carries the critic's flagged/unverified clips, so `data/videos/` is self-describing without re-reading the run log.

**Clip reuse is backend-gated too** (`PROVENANCE_FILE` in `visual_gen.py`), and this is a *separate* check from the manifest above — they guard different things and you need both. The manifest decides whether an episode is skipped entirely; `.visual_backend`, written inside `data/assets/<video_id>/`, decides whether individual clips already in that directory may be reused. Skipping the manifest check but not this one means a Veo run resuming into a directory left by a Pillow pass silently reuses the stills, calls Veo zero times, reports success — and then the batch runner writes a manifest claiming the episode *was* rendered with Veo, so the lie persists and every later run skips it.

| state of `data/assets/<video_id>/` | action |
|---|---|
| marker matches the current backend | reuse existing clips |
| marker names a different backend | regenerate, warning that the clips don't match |
| clips present, no marker (predates this check) | regenerate — provenance unknown, so don't assume |
| empty directory | nothing to reuse; nothing to warn about |

The marker is written at the **start** of a run, not the end, so an interrupted run still resumes into its own clips rather than treating them as unknown-provenance. `--regenerate` (i.e. `reuse_existing=False`) overrides a matching marker.

Sequential on purpose: Veo is rate-limited, and a batch you can Ctrl-C without leaving half-written parallel state is worth more than the wall-clock a concurrent version would save.

## Continuity critic (`visuals/critic.py`)

The pipeline's feedback loop. Without it, `visual_gen` generates and `assembly` muxes without anything ever *looking* at the output — a swapped character surfaces only when a human watches the finished episode, which is exactly what an unattended batch doesn't do.

After each clip is generated, frames are sampled and sent to Claude alongside the hero reference the clip was anchored to, with one question: is this still the same scene with the same characters? Opt-in per strategy via `visuals.critic: true`; `--no-critic` disables for a run.

**Why it runs inline, not as a pass at the end.** A bad clip becomes the chain anchor for the next one. A corrupted clip 1 silently poisons clips 2 and 3, so a post-hoc pass would flag three clips where an inline check prevents two of them from ever being generated wrong.

**Severity, not a boolean.** Identity failures (character replaced/missing/swapped, new character, setting changed) are `fatal` and trigger a regenerate. Generational softening, set-dressing drift, and minor camera movement are `minor` — reported, but not worth paying to regenerate. Verified in practice: a known-good clip was correctly flagged `minor: camera_moved` for a slight push-in and still passed.

**Escalation ladder**, mirroring the existing safety-filter retry: (1) generate; (2) on a fatal verdict regenerate unchanged, since the failures are statistical; (3) still fatal — drop `chain_from` and re-anchor to the hero, because chaining from a poisoned frame just reproduces the fault. Exhausted: keep the last clip and log `FLAGGED FOR REVIEW`. A flagged clip in a finished episode beats a failed render.

**Chained clips are judged against the segment's hero, not the frame they chained from** — otherwise a drifting clip gets compared against the drift and passes.

**It can never block a render.** Missing key, network error, refusal, malformed response — all degrade to a passing verdict marked `unverified`, reported separately at the end so "never checked" can't be mistaken for "checked and fine". A critic that crashes a batch is worse than no critic.

**Bias toward passing.** A false positive costs one extra clip; a false negative costs a broken episode. But a critic that flags everything burns the budget and trains you to ignore it, so the prompt says explicitly: flag clear failures, and when uncertain, pass.

**The critic is told the shot's intended action** (`Segment.staged_action`, joined from the turns' stage directions). The reference image is a single frozen frame from the *start* of the shot, so without this a scripted exit — "PERE UBU going out, slamming the door" — is indistinguishable from Veo losing the character, and gets scored `fatal: character_missing`. That failure can never be fixed by regenerating, so it burns the entire 3-attempt ladder every time and keeps a clip it still considers broken. Found in Ubu Rex e03 segment 1: three clips × three attempts = nine Veo calls, all flagged, on shots that were doing exactly what the script asked. Single-narrator strategies carry no turns and so send no action, leaving their prompt unchanged.

The exemption is deliberately narrow, because "the character left" is an excuse that would otherwise cover a completely broken shot. A `blank_frames` kind was added alongside it so an empty or black clip has its own fatal channel rather than having to arrive as `character_missing` — the one verdict the exemption now waves through — and the prompt states outright that a scripted exit never explains a vanished setting or every character disappearing at once.

**A file's extension is not evidence of its format.** `detect_media_type` sniffs magic bytes (PNG/JPEG/GIF/WEBP, defaulting to PNG) rather than trusting the suffix, because reference images written straight from an image API's response bytes routinely carry a `.png` name while actually being JPEG. The old hardcoded `image/png` made the API reject the request with a 400 that reads like a code bug rather than a data one. `scripts/generate_refs.py` prints a note when the edit API hands back a type that doesn't match the manifest's filename — the content is fine, the name just lies.

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
- ~~**Whisper subtitle alignment**~~ — done. `src/shortform/tts/whisper_align.py` recovers word timings for backends that don't emit them (F5-TTS), gated on `visuals.subtitles`. Soft-deps on `faster-whisper` via the `captions` extra; degrades to no captions if absent. It's free transcription rather than forced alignment against the known script, so caption words come from ASR and can occasionally differ from the narration — upgrade to whisperx/stable-ts if that drift ever matters.
- **Alternate visual backends** — Kling for looser-filter content, or another local-model attempt on heavier hardware. The `src/shortform/visuals/registry.py` pattern makes adding one straightforward.

## Key decisions already made (don't re-debate without new info)

- **Python, not TypeScript** — better ML/video ecosystem.
- **Local F5-TTS over hosted ElevenLabs** — free, high quality, no character limits.
- **Veo image-to-video, not text-to-video** — stronger consistency, the reference-image anchor extends naturally to the variant system.
- **Linear stages with SQLite checkpointing** — solid; improvements happen at the *content* layer, not the orchestration layer.
- **Claude for script generation, not Gemini** — quality difference matters for tone-sensitive writing.
- **No publishing automation yet** — Phase 2.
- **One repo, multiple strategies** — keeping motivation_quotes and tech_tips alongside gothic_vignette makes the pipeline's flexibility legible.
