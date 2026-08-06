"""Generate character/scene reference images from a variants manifest.

The manifest-driven generalization of scripts/generate_bartholomew.py (which
stays as-is — it carries Bartholomew's canonical prompt inline). Point this at
any manifest that declares a `hero` prompt and a list of `variants`:

    # 1. Generate candidates for the anchor image and eyeball them
    uv run python scripts/generate_refs.py data/character_refs/<name>/variants.yaml --candidates 4

    # 2. Lock the one you want
    uv run python scripts/generate_refs.py data/character_refs/<name>/variants.yaml \\
        --lock data/character_refs/<name>/candidates/cand_00.png

    # 3. Derive every variant by EDITING the locked anchor
    uv run python scripts/generate_refs.py data/character_refs/<name>/variants.yaml --variants
    uv run python scripts/generate_refs.py <manifest> --variants --only <key> --force

Why the lock-then-edit order matters: image EDITING preserves identity far
better than text-to-image regeneration. For a multi-character series the anchor
frame must contain every principal who appears together, because that one frame
is what every later variant inherits its faces from.

Costs (approximate, verify current pricing):
    Imagen 4 standard  ~$0.04/image  -> 4 candidates ~ $0.16
    Nano Banana Pro    ~$0.04/image  -> 1 variant edit ~ $0.04
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGEN_MODEL_DEFAULT = "imagen-4.0-generate-001"
EDIT_MODEL_DEFAULT = "nano-banana-pro-preview"
IMAGEN_MAX_PER_CALL = 4  # Imagen 4 caps number_of_images per request


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"manifest not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not data.get("hero", {}).get("prompt"):
        raise SystemExit(f"{path}: manifest needs hero.prompt")
    return data


def manifest_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    """(base_dir, locked_hero_path, candidates_dir) resolved against the repo."""
    base = PROJECT_ROOT / str(manifest.get("base_dir", "data/character_refs"))
    hero = base / str(manifest.get("locked_hero", "hero.png"))
    return base, hero, base / "candidates"


def generate_candidates(
    api_key: str, prompt: str, count: int, out_dir: Path, model: str
) -> list[Path]:
    client = genai.Client(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved: list[Path] = []
    remaining, batch = count, 0
    while remaining > 0:
        size = min(remaining, IMAGEN_MAX_PER_CALL)
        print(f"Calling Imagen ({model}) for {size} candidate(s)...")
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=size,
                aspect_ratio="9:16",
                person_generation="allow_adult",
            ),
        )
        if not response.generated_images:
            raise SystemExit(
                "Imagen returned no images — almost certainly a safety-filter "
                "rejection. Soften the prompt and retry."
            )
        for i, gen in enumerate(response.generated_images):
            path = out_dir / f"cand_{stamp}_b{batch}_{i:02d}.png"
            gen.image.save(str(path))
            saved.append(path)
            print(f"  saved {path.relative_to(PROJECT_ROOT)}")
        remaining -= size
        batch += 1
    return saved


def edit_variant(
    api_key: str, base_image: Path, edit_prompt: str, out_path: Path, model: str
) -> None:
    """Derive one variant by editing the locked anchor image."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=base_image.read_bytes(), mime_type="image/png"),
            types.Part.from_text(text=edit_prompt),
        ],
    )
    if not response.candidates:
        raise RuntimeError(f"no candidates returned for {out_path.name}")

    parts = response.candidates[0].content.parts or []
    image_part = next((p for p in parts if getattr(p, "inline_data", None)), None)
    if image_part is None:
        blob = " ".join((getattr(p, "text", "") or "") for p in parts)[:300]
        raise RuntimeError(f"no image in response for {out_path.name}; text: {blob!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(image_part.inline_data.data)


def run_variants(
    api_key: str,
    manifest: dict[str, Any],
    base: Path,
    hero: Path,
    only: str | None,
    force: bool,
    model: str,
) -> int:
    if not hero.exists():
        raise SystemExit(
            f"locked anchor not found: {hero}\n"
            f"  Generate candidates first (--candidates N), then --lock one."
        )

    targets = [
        v
        for v in manifest.get("variants", [])
        if v.get("edit_prompt") and (not only or v.get("key") == only)
    ]
    if not targets:
        print(f"No variants to edit{f' matching {only!r}' if only else ''}.")
        return 0

    print(f"Editing {len(targets)} variant(s) via {model}")
    print(f"Estimated cost: ~${0.04 * len(targets):.2f}\n")

    failures = 0
    for v in targets:
        out_path = base / str(v["file"])
        key = str(v["key"])
        if out_path.exists() and not force:
            print(f"  [skip] {key} — exists (use --force to regenerate)")
            continue
        print(f"  [edit] {key} -> {out_path.relative_to(PROJECT_ROOT)}")
        try:
            edit_variant(api_key, hero, str(v["edit_prompt"]), out_path, model)
            print(f"  [ok]   {key}")
        except RuntimeError as e:
            print(f"  [FAIL] {key} — {e}")
            failures += 1
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("manifest", type=Path, help="path to a variants manifest YAML")
    ap.add_argument("--candidates", type=int, metavar="N", help="generate N anchor candidates")
    ap.add_argument("--lock", type=Path, metavar="PNG", help="lock this candidate as the anchor")
    ap.add_argument("--variants", action="store_true", help="edit-derive all variants")
    ap.add_argument("--only", help="with --variants, only this key")
    ap.add_argument("--force", action="store_true", help="regenerate existing files")
    ap.add_argument("--imagen-model", default=IMAGEN_MODEL_DEFAULT)
    ap.add_argument("--edit-model", default=EDIT_MODEL_DEFAULT)
    args = ap.parse_args()

    if not (args.candidates or args.lock or args.variants):
        ap.error("pick one of --candidates N, --lock PNG, or --variants")

    manifest = load_manifest(args.manifest)
    base, hero, candidates_dir = manifest_paths(manifest)

    # --lock is pure file movement; no API key needed.
    if args.lock:
        src = args.lock if args.lock.is_absolute() else PROJECT_ROOT / args.lock
        if not src.exists():
            raise SystemExit(f"candidate not found: {src}")
        hero.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, hero)
        print(f"Locked anchor: {src.name} -> {hero.relative_to(PROJECT_ROOT)}")
        if not (args.candidates or args.variants):
            return 0

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GOOGLE_GEMINI_API_KEY not set (see .env)")

    if args.candidates:
        print(f"Estimated cost: ~${0.04 * args.candidates:.2f}\n")
        saved = generate_candidates(
            api_key, str(manifest["hero"]["prompt"]), args.candidates,
            candidates_dir, args.imagen_model,
        )
        print(f"\n{len(saved)} candidate(s) in {candidates_dir.relative_to(PROJECT_ROOT)}/")
        print("Review them, then lock one:")
        print(f"  uv run python scripts/generate_refs.py {args.manifest} --lock <path>")

    if args.variants:
        failures = run_variants(
            api_key, manifest, base, hero, args.only, args.force, args.edit_model
        )
        if failures:
            print(f"\n{failures} variant(s) failed.", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
