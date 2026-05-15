"""Generate Bartholomew character reference image candidates via Imagen 4,
plus per-scene variants via Nano Banana Pro (Gemini 3 Pro Image).

Bartholomew is the clay-skeleton protagonist of the gothic_vignette strategy.
This script has two modes:

(1) Generate new hero candidates from scratch (Imagen 4). For when you want
    to iterate on the locked hero. Outputs to data/character_refs/candidates/.

    uv run python scripts/generate_bartholomew.py            # 6 candidates
    uv run python scripts/generate_bartholomew.py --count 4
    uv run python scripts/generate_bartholomew.py --variant standing

(2) Generate scene-variant hero PNGs from the existing locked hero via
    image editing (Nano Banana Pro). Each variant in
    data/character_refs/variants.yaml that has an edit_prompt gets rendered.
    Preserves character identity across scene changes.

    uv run python scripts/generate_bartholomew.py --edit-variants
    uv run python scripts/generate_bartholomew.py --edit-variants --only kitchen
    uv run python scripts/generate_bartholomew.py --edit-variants --force

Costs:
    Imagen 4 standard: ~$0.04/image. 6 candidates ~ $0.24.
    Nano Banana Pro:   ~$0.04/image. 5 variant edits ~ $0.20.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHARACTER_REFS_DIR = PROJECT_ROOT / "data" / "character_refs"
CANDIDATES_DIR = CHARACTER_REFS_DIR / "candidates"
VARIANTS_DIR = CHARACTER_REFS_DIR / "variants"
VARIANTS_MANIFEST = CHARACTER_REFS_DIR / "variants.yaml"
LOCKED_HERO = CHARACTER_REFS_DIR / "bartholomew_hero.png"

EDIT_MODEL_DEFAULT = "nano-banana-pro-preview"

# The canonical Bartholomew prompt — anchors the entire series.
# Avoid naming Tim Burton directly (Imagen sometimes interprets as the person).
# Style descriptors instead: claymation, stop-motion, gothic whimsy.
BASE_PROMPT = """\
A character portrait of a humble clay skeleton — hand-sculpted plasticine, \
visible fingerprints and tool marks, in the aesthetic of high-budget gothic \
stop-motion animation (the lineage of Corpse Bride and Frankenweenie). \
The skeleton wears a moth-eaten chocolate-brown tweed cardigan over a faded \
ivory shirt with an oversized crooked burgundy bowtie. Round wire-rim \
spectacles perch precariously on the skull. A small dented bowler hat sits \
slightly askew. He is seated at a small wooden writing desk in a dim \
Victorian study, skeletal hands resting on its edge, gazing contemplatively \
slightly off-camera. Behind him: a sliver of mahogany bookshelf, a single \
lit beeswax candle, an anachronistic small cathode-ray computer monitor, \
a ceramic funerary urn beside a stack of yellowed unpaid bills.

Lighting: soft warm candlelight from camera-left, deep umber shadows, \
chiaroscuro. Composition: vertical portrait, medium shot from waist up, \
shallow depth of field with background gently out of focus, 35mm-equivalent \
lens. Palette: sepia, oxblood, deep umber, aged ivory — desaturated and \
muted. Mood: melancholic, quietly absurd, gothic whimsy.

Negative: no human face, no flesh, no modern photography style, no text, \
no watermarks, no cartoon style, no live-action realism, no bright \
saturated colors, no daylight."""


VARIANTS = {
    "default": BASE_PROMPT,
    "no_hat": BASE_PROMPT.replace(
        "A small dented bowler hat sits slightly askew. ", ""
    ),
    "wider": BASE_PROMPT.replace(
        "medium shot from waist up", "medium-wide shot showing the full desk and study"
    ),
    "closer": BASE_PROMPT.replace(
        "medium shot from waist up", "tight portrait, head and shoulders, the skull dominating the frame"
    ),
    "standing": BASE_PROMPT.replace(
        "He is seated at a small wooden writing desk in a dim Victorian study, "
        "skeletal hands resting on its edge, gazing contemplatively slightly off-camera. ",
        "He stands beside a small wooden writing desk in a dim Victorian study, "
        "one skeletal hand resting on its surface, gazing contemplatively at "
        "something just out of frame. ",
    ),
}


IMAGEN_MAX_PER_CALL = 4  # Imagen 4 API caps number_of_images at 4 per request


def generate(
    api_key: str,
    prompt: str,
    count: int,
    output_dir: Path,
    variant_name: str,
    model: str = "imagen-4.0-generate-001",
) -> list[Path]:
    """Generate `count` images and save them to output_dir. Returns saved paths.

    Batches into multiple API calls when count > IMAGEN_MAX_PER_CALL.
    """
    client = genai.Client(api_key=api_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: list[Path] = []
    remaining = count
    batch_index = 0

    while remaining > 0:
        batch_size = min(remaining, IMAGEN_MAX_PER_CALL)
        print(f"Calling Imagen ({model}) for {batch_size} candidate(s)...")
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=batch_size,
                aspect_ratio="9:16",
                person_generation="allow_adult",
            ),
        )

        if not response.generated_images:
            print("ERROR: Imagen returned no images. Likely a safety filter rejection.")
            sys.exit(1)

        for i, gen in enumerate(response.generated_images):
            path = output_dir / f"bartholomew_{variant_name}_{timestamp}_b{batch_index}_{i:02d}.png"
            gen.image.save(str(path))
            saved.append(path)
            print(f"  saved {path.relative_to(PROJECT_ROOT)}")

        remaining -= batch_size
        batch_index += 1

    return saved


def edit_variant(
    api_key: str,
    base_image_path: Path,
    edit_prompt: str,
    output_path: Path,
    model: str = EDIT_MODEL_DEFAULT,
) -> None:
    """Generate one scene-variant by editing the locked hero PNG via Nano Banana Pro.

    The edit prompt should describe the target scene while preserving the
    character. See variants.yaml for canonical phrasing.
    """
    client = genai.Client(api_key=api_key)

    image_bytes = base_image_path.read_bytes()
    base_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    prompt_part = types.Part.from_text(text=edit_prompt)

    response = client.models.generate_content(
        model=model,
        contents=[base_part, prompt_part],
    )

    if not response.candidates:
        raise RuntimeError(f"No candidates returned for {output_path.name}")

    parts = response.candidates[0].content.parts or []
    image_part = next(
        (p for p in parts if getattr(p, "inline_data", None) is not None),
        None,
    )
    if image_part is None:
        text_blob = " ".join(
            (getattr(p, "text", "") or "") for p in parts
        )[:300]
        raise RuntimeError(
            f"No image in response for {output_path.name}. "
            f"Text fragments: {text_blob!r}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(image_part.inline_data.data)


def run_edit_variants(
    api_key: str,
    manifest_path: Path,
    only: str | None,
    force: bool,
    model: str,
) -> None:
    if not manifest_path.exists():
        print(f"ERROR: variants manifest not found at {manifest_path}")
        sys.exit(1)
    if not LOCKED_HERO.exists():
        print(f"ERROR: locked hero not found at {LOCKED_HERO}")
        sys.exit(1)

    data = yaml.safe_load(manifest_path.read_text()) or {}
    variants = data.get("variants", [])

    targets: list[dict] = []
    for v in variants:
        if not v.get("edit_prompt"):
            continue
        if only and v["key"] != only:
            continue
        targets.append(v)

    if not targets:
        if only:
            print(f"No variant matching --only '{only}' (or it has no edit_prompt).")
        else:
            print("No variants have edit_prompt set — nothing to do.")
        return

    estimated_cost = 0.04 * len(targets)
    print(f"Editing {len(targets)} variant(s) via {model}")
    print(f"Estimated cost: ~${estimated_cost:.2f}\n")

    for v in targets:
        output_path = CHARACTER_REFS_DIR / v["file"]
        if output_path.exists() and not force:
            print(f"  [skip] {v['key']:8} — {output_path.relative_to(PROJECT_ROOT)} exists (use --force to regenerate)")
            continue
        print(f"  [edit] {v['key']:8} → {output_path.relative_to(PROJECT_ROOT)}")
        try:
            edit_variant(
                api_key=api_key,
                base_image_path=LOCKED_HERO,
                edit_prompt=v["edit_prompt"],
                output_path=output_path,
                model=model,
            )
            print(f"  [ok]   {v['key']:8} — saved")
        except RuntimeError as e:
            print(f"  [FAIL] {v['key']:8} — {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=6, help="Number of candidates to generate (Imagen mode)")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="default",
        help="Prompt variant for Imagen mode — different framing/pose/composition",
    )
    parser.add_argument(
        "--model",
        default="imagen-4.0-generate-001",
        help="Imagen model id for new-hero generation (default: imagen-4.0-generate-001)",
    )

    # Edit-variants mode
    parser.add_argument(
        "--edit-variants", action="store_true",
        help="Switch to variant-edit mode: produce scene variants by editing the locked hero.",
    )
    parser.add_argument(
        "--only", default=None,
        help="In edit mode, render only the variant with this key.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="In edit mode, overwrite existing variant files.",
    )
    parser.add_argument(
        "--edit-model", default=EDIT_MODEL_DEFAULT,
        help=f"Image-edit model for variant mode (default: {EDIT_MODEL_DEFAULT})",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_GEMINI_API_KEY not set in .env")
        sys.exit(1)

    if args.edit_variants:
        run_edit_variants(
            api_key=api_key,
            manifest_path=VARIANTS_MANIFEST,
            only=args.only,
            force=args.force,
            model=args.edit_model,
        )
        return

    prompt = VARIANTS[args.variant]
    print(f"Variant: {args.variant}")
    print(f"Estimated cost: ~${0.04 * args.count:.2f} (at $0.04/image for Imagen 4 standard)\n")

    saved = generate(
        api_key=api_key,
        prompt=prompt,
        count=args.count,
        output_dir=CANDIDATES_DIR,
        variant_name=args.variant,
        model=args.model,
    )

    print(f"\nGenerated {len(saved)} candidate(s) in {CANDIDATES_DIR.relative_to(PROJECT_ROOT)}/")
    print("\nNext steps:")
    print("  1. Browse the candidates and pick your favorite")
    print("  2. Copy/rename your pick to data/character_refs/bartholomew_hero.png")
    print("     cp data/character_refs/candidates/<your_pick>.png data/character_refs/bartholomew_hero.png")
    print("  3. The gothic_vignette strategy will pick it up automatically on the next pipeline run")
    print("\nNot loving any of them? Try a different variant:")
    print(f"  uv run python scripts/generate_bartholomew.py --variant {next(iter([v for v in VARIANTS if v != args.variant]))}")


if __name__ == "__main__":
    main()
