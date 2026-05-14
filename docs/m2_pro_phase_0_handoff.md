# M2 Pro Phase 0 Handoff: Prove HunyuanVideo I2V Runs Locally

**Audience.** A Claude Code session running on the user's M2 Pro MacBook (the "spare" machine in CLAUDE.md's Machine Topology section, which we're about to un-retire).

**Author.** Claude on the user's M4 Air, after a long session on 2026-05-13 to 14 in which we landed the entire Bartholomew pipeline end-to-end (Veo + F5-TTS + multi-clip + chaining + first YouTube upload).

**Why this doc exists.** We hit a wall with Veo's pricing + safety filters. The user is dropping ~$5/video and burning through prepaid credits in days, AND Veo's safety filter rejects ~30% of segments containing skeletal-language prompts (which is fundamental to Bartholomew). We agreed the long-term fix is a **local open-source video model on the M2 Pro**, leaving the M4 Air for code, watching outputs, and orchestrating. Your job in Phase 0 is to **prove HunyuanVideo I2V actually runs on this M2 Pro and produces a usable Bartholomew clip**. Nothing more. Don't integrate, don't modify the project code, don't optimize. Phase 0 is a gate.

---

## What you should already have from the repo

After the user runs `git pull` on this M2 Pro, the working tree should be at commit `cdc363c` or later. Key files for context:

- `CLAUDE.md` — project context. **Read this first if you haven't.** The "F5-TTS Setup Notes" section is the closest analog to what you're about to set up.
- `src/shortform/visuals/veo_backend.py` — the working backend you're eventually going to mirror (NOT TONIGHT). Useful as a reference for the integration shape.
- `data/character_refs/bartholomew_hero.png` — the locked Bartholomew hero reference. This is the image you'll feed to HunyuanVideo I2V for the test generation. ~9:16 aspect ratio, 1080x1920, hand-curated from Imagen.
- `config/strategies/gothic_vignette.yaml` — the strategy. Contains the canonical visual prompt template + the six Bartholomew few-shot examples. Look at the `animation_style` field for the kind of prompt you'll use in the test.

---

## What state the M2 Pro might be in (verify first)

We did F5-TTS PoC work on this machine roughly 2026-05-11 to 12 in a separate Claude session. As of that work:

- **F5-TTS installed** at `~/.venvs/f5-tts/` (uv-managed Python 3.12 venv with `f5-tts` from PyPI)
- **Voice reference** at `data/voices/bartholomew_reference.m4a` + `bartholomew_reference_trimmed.wav`. These are gitignored — they may or may not still exist on the Pro depending on whether the user copied them off.
- **Repo** may or may not be cloned here. The user's been on the Air primarily.

**Before doing anything else: actually check the state.**

```bash
# Where's the user's home?
echo $HOME

# Is the repo here?
ls ~/WebstormProjects/short_form_test_runner 2>/dev/null && echo "REPO PRESENT" || echo "NO REPO"

# Is F5-TTS venv still here?
ls ~/.venvs/f5-tts/bin/f5-tts_infer-cli 2>/dev/null && echo "F5-TTS PRESENT" || echo "NO F5-TTS"

# Available disk space (need ~30GB free for ComfyUI + HunyuanVideo I2V weights)
df -h ~

# Memory + chip confirmation
system_profiler SPHardwareDataType | grep -E "Model|Chip|Memory"

# Python + uv
python3 --version
which uv && uv --version || echo "NO UV"
ffmpeg -version 2>&1 | head -1 || echo "NO FFMPEG"
```

**Expected:** Apple M2 Pro, 32GB memory, recent macOS. uv + ffmpeg + Homebrew should be installed.

If the repo isn't here, clone it: `git clone https://github.com/iaj6/short_form_test_runner ~/WebstormProjects/short_form_test_runner` (private repo — auth as needed).

---

## The mission (Phase 0 only)

**Generate ONE 5-second image-to-video clip from `data/character_refs/bartholomew_hero.png` using HunyuanVideo I2V locally on this M2 Pro, animated with a Bartholomew-appropriate prompt.** Watch the result with the user. Decide together if quality is acceptable.

**Out of scope for Phase 0 (do not do these):**
- Integrating with the project pipeline (no Python module, no FastAPI service, no backend file)
- Building multi-clip or chaining workflows
- Setting up any HTTP server
- Modifying anything in `src/`
- Optimizing inference speed

The goal is a **single MP4 file you can show the user** so they can answer: "Does this look like Bartholomew, and is the quality acceptable enough to keep going?"

---

## Recommended approach: ComfyUI with HunyuanVideo nodes

There are two reasonable ways to run HunyuanVideo on Apple Silicon:

1. **ComfyUI + HunyuanVideoWrapper custom nodes** — node-based workflow editor. Big community, lots of pre-built workflows, easy to swap models. **Recommended for Phase 0** because the iteration loop is fastest.
2. **Diffusers library + raw Python script** — leaner, no GUI. More code to write. Recommended for production but overkill for a PoC.

Go with ComfyUI. Steps below.

### Step 1: Install ComfyUI

```bash
# Pick a directory for ComfyUI install (not inside the project repo)
mkdir -p ~/tools && cd ~/tools

git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create a dedicated venv (don't reuse the F5-TTS one — different deps)
uv venv .venv --python 3.12
source .venv/bin/activate

# Install ComfyUI's Python deps. The requirements file uses CUDA torch by
# default; for Apple Silicon you want the MPS-enabled torch build.
uv pip install --upgrade pip
uv pip install torch torchvision torchaudio  # picks the right wheel for ARM macOS
uv pip install -r requirements.txt

# Confirm MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

If MPS reports `False`, stop and report back — that breaks the whole plan and we need to figure out why.

### Step 2: Install HunyuanVideo wrapper nodes

ComfyUI supports HunyuanVideo via the `ComfyUI-HunyuanVideoWrapper` custom node pack maintained by Kijai (the de facto standard for these wrappers).

```bash
cd ~/tools/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-HunyuanVideoWrapper.git
cd ComfyUI-HunyuanVideoWrapper
uv pip install -r requirements.txt
```

There may be additional helper nodes (`ComfyUI-VideoHelperSuite` is commonly needed for video I/O). Install if the workflow we use later requires them — the workflow JSON will tell ComfyUI which nodes are needed when loaded.

### Step 3: Download HunyuanVideo I2V weights

You need the **image-to-video variant**, not the text-to-video one. The I2V checkpoint is a separate file.

Standard location: `~/tools/ComfyUI/models/diffusion_models/` for the main model, `models/vae/` for the VAE, `models/text_encoders/` for the text encoders (CLIP + LLaVA-style).

```bash
cd ~/tools/ComfyUI/models

# These are the canonical paths Kijai's wrapper expects. Use hf-cli or wget.
# Total download is ~20-25GB. One-time.

# Main I2V transformer (the big one, ~13GB)
huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_I2V_720_fp8_e4m3fn.safetensors \
  --local-dir diffusion_models

# VAE (~250MB)
huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_bf16.safetensors \
  --local-dir vae

# Text encoders (~6-8GB combined)
huggingface-cli download Kijai/llava-llama-3-8b-text-encoder-tokenizer \
  --local-dir text_encoders/llava-llama-3-8b-text-encoder-tokenizer
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir text_encoders/clip-vit-large-patch14
```

The `fp8` variant of the main model is quantized to 8-bit floats — it fits in less memory and runs faster on Apple Silicon than the full BF16 version. 32GB of unified memory should handle it.

If `huggingface-cli` isn't installed: `uv pip install huggingface_hub[cli]`. You may also need `huggingface-cli login` if any of the repos are gated.

### Step 4: Start ComfyUI

```bash
cd ~/tools/ComfyUI
source .venv/bin/activate
python main.py --listen 0.0.0.0
```

`--listen 0.0.0.0` exposes ComfyUI on the LAN (useful later for the integration, harmless now). Default port is 8188. Open `http://localhost:8188` in a browser.

If it crashes at startup with MPS-related errors, try:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python main.py --listen 0.0.0.0
```

### Step 5: Load an I2V workflow

In the ComfyUI web UI, you need an image-to-video workflow targeting HunyuanVideo. Three ways to get one:

1. **Kijai's example workflows** — the HunyuanVideoWrapper repo has `example_workflows/` with JSON files you can drag-and-drop onto the ComfyUI canvas. Look for one named like `hunyuan_video_image_to_video.json` or similar.
2. **CivitAI** — search "HunyuanVideo I2V" for community workflows.
3. **Build minimally** — load image, HunyuanVideo I2V sampler, decode, save as video.

Drop in the workflow, then configure the nodes:
- **Input image**: `~/WebstormProjects/short_form_test_runner/data/character_refs/bartholomew_hero.png` (you may need to copy it into `ComfyUI/input/` or use an absolute path node)
- **Prompt**: Use something close to what the gothic strategy actually feeds Veo. Example for the test:
  > `stop-motion claymation, 24fps on twos, slow push-in, soft candlelight, shallow depth of field, muted gothic palette, Tim Burton aesthetic, the clay skeleton character sits at his Victorian writing desk, reaching for a quill, jaw clacking shut`
- **Negative prompt**: `low quality, blurry, distorted, deformed, modern photography, photorealistic`
- **Resolution**: 720x1280 (closest aspect to your 1080x1920 target that the model handles well; you can upscale in post)
- **Duration / frame count**: 5 seconds = ~120 frames at 24fps. Or whatever the workflow's default is — don't push past the model's max.
- **Steps**: 30-50 is typical. More = slower + slightly better quality.
- **Seed**: any fixed integer so you can re-run reproducibly if needed.

### Step 6: Run the generation

Click "Queue Prompt." This will take a long time. **Expect 20-45 minutes for the first 5-second clip** on M2 Pro:

- Model loading from disk: 2-5 min (one-time per ComfyUI session — cached after)
- Inference: 15-40 min
- VAE decode + save: 1-2 min

The fans will spin. The machine will be hot. This is normal. If it OOMs (out-of-memory error in the console output), the fp8 model variant isn't fitting — try lowering frame count, reducing resolution, or switching to an even-more-quantized model.

While it runs, save the output WAV path so you can find it: ComfyUI typically saves to `~/tools/ComfyUI/output/`.

### Step 7: Watch the output and judge

Open the resulting MP4. Sit with the user. Ask:

1. **Is this Bartholomew?** Does the character preserve from the reference image — the moth-eaten tweed, the round spectacles, the bowler hat? Does the Victorian setting come through?
2. **Is the motion claymation-y?** Stop-motion-on-twos? Or did it produce smooth CG?
3. **Is the quality acceptable for Shorts?** Or is it artifacty/blurry in ways Veo's output isn't?

**Decision gate.**
- **YES across the board (or close enough):** Phase 0 passes. Report back. Air-side Claude will plan Phase 1 (side-by-side quality comparison vs Veo).
- **Mostly yes, but some weakness (motion is off, character drifts, etc.):** Phase 0 conditionally passes. Save the output, note what's weak, report back. Air-side will weigh whether to push forward or try Wan 2.2 instead.
- **NO — quality is unusable:** Phase 0 fails. Report back. Options at that point are (a) try Wan 2.2 instead, (b) drop the local-model idea and look at Kling's API as the cheaper-than-Veo path.

---

## What to report back to the user (and the Air-side Claude)

After running, please record:

1. **Wall time** — how long did the inference actually take?
2. **Peak memory usage** — `vm_stat` or Activity Monitor while it's running. Did the M2 Pro stay under 32GB or did it swap?
3. **Output file path** — where did the MP4 land?
4. **Subjective quality notes** — your honest read on the result
5. **Any errors / fallbacks** — did MPS work cleanly or did anything fall back to CPU?
6. **The exact workflow JSON you ended up using** — paste it into a new file at `~/WebstormProjects/short_form_test_runner/docs/m2_pro_phase_0_workflow.json` so Air-side Claude can see exactly what params produced this output. Commit it.

Either commit your notes as a sibling markdown file (`docs/m2_pro_phase_0_results.md`) and push, or paste them to the user so they bring them back to the Air session.

---

## Things to NOT do (scope discipline)

- **Don't** modify any file under `src/`. Phase 0 is an external-tooling test, not a project integration.
- **Don't** write Python code that imports anything from `shortform.*`. That comes in Phase 2.
- **Don't** try to chain multiple clips or build any kind of pipeline. One clip, one watch, one decision.
- **Don't** install ComfyUI inside the project repo — keep it at `~/tools/ComfyUI` so the project venv stays clean.
- **Don't** burn time trying to get *perfect* output. Phase 0 is "is this viable?" not "is this production-ready?"
- **Don't** delete or touch the `~/.venvs/f5-tts/` install. Even if F5-TTS is unused for Phase 0, it'll be needed in Phase 3.

---

## Quick reference: the M4 Air side's current state

So you know what you're handing back to:

- Pipeline works end-to-end via Veo: ~25 min wall, ~$5 Veo per video
- 6 finished Bartholomew videos exist at `data/videos/`
- First Short is publicly live: https://www.youtube.com/channel/UCH0aAvB6yTWic68j2Q0gsvg
- `src/shortform/visuals/veo_backend.py` is the file to mirror eventually (Phase 2 work)
- Strategy YAML at `config/strategies/gothic_vignette.yaml` is where the new backend would get wired in via `visuals.backend: "local_video"` once it exists
- F5-TTS integration is at `src/shortform/tts/f5_backend.py` (currently subprocess-style; will move to HTTP in Phase 3)

---

## If you get stuck

The two most likely failure modes:

1. **MPS-related crash or extreme slowness.** PyTorch's MPS backend on Apple Silicon has been improving but still has rough edges. If specific ops fall back to CPU and inference takes 4+ hours per clip, the practical answer might be: this approach isn't viable on this machine right now, try Wan 2.2 (different ops, may behave better) or wait for better MPS support.
2. **OOM with the fp8 model.** Try the int8-quantized variants if Kijai provides them, or use a smaller frame count for the test (3 seconds instead of 5).

If you hit something that doesn't fit either of those, write up the symptom + what you tried and the user will bring it back to me on the Air side. Don't grind for hours.
