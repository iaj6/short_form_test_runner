"""Generate Bartholomew character reference image candidates via Imagen 4.

Bartholomew is the clay-skeleton protagonist of the gothic_vignette strategy.
This script produces N candidate hero images saved to
data/character_refs/candidates/. Browse them, pick the one that nails the
character, and copy/rename your pick to data/character_refs/bartholomew_hero.png
— that's the path the gothic_vignette strategy reads as the Veo base frame.

Usage:
    uv run python scripts/generate_bartholomew.py            # 6 candidates
    uv run python scripts/generate_bartholomew.py --count 4
    uv run python scripts/generate_bartholomew.py --variant pose

Cost: ~$0.04/image with Imagen 4 standard. 6 candidates ~ $0.24.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_DIR = PROJECT_ROOT / "data" / "character_refs" / "candidates"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=6, help="Number of candidates to generate")
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()),
        default="default",
        help="Prompt variant — different framing/pose/composition",
    )
    parser.add_argument(
        "--model",
        default="imagen-4.0-generate-001",
        help="Imagen model id (default: imagen-4.0-generate-001)",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_GEMINI_API_KEY not set in .env")
        sys.exit(1)

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
