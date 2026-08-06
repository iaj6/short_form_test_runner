"""Generate music cues declared in a category's tracks.yaml via ElevenLabs.

    uv run python scripts/generate_music.py <category>
    uv run python scripts/generate_music.py <category> --dry-run
    uv run python scripts/generate_music.py <category> --only <file-stem> --force

Manifest-driven, like scripts/generate_refs.py: any track entry carrying a
`prompt` and lacking its audio file gets generated. The manifest is checked into
git and the audio is not, so a cue set is reproducible from the repo — which is
also why the prompt belongs in the manifest rather than in this script.

GENERATE A SERIES SCORE, NOT PER-EPISODE MUSIC. It's tempting to generate a
fresh track for every episode, but recurring cues are a large part of why a
serial sounds like one show, and a fixed set is cheaper. Write a handful of
cues with distinct moods and let the selector match them per episode.

Prompts should describe an aesthetic — instrumentation, tempo, key, feel — not
name a specific existing work.

Requires ELEVENLABS_API_KEY. Cost scales with requested length; check current
pricing before generating a large set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MUSIC_ROOT = PROJECT_ROOT / "data" / "music"
API_URL = "https://api.elevenlabs.io/v1/music"

# Long enough that the loop seam isn't obvious under a 60-90s episode, short
# enough not to pay for audio nobody hears. Assembly loops to length anyway.
DEFAULT_LENGTH_MS = 45_000
REQUEST_TIMEOUT_SECONDS = 300


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"manifest not found: {path}\n"
            f"  Create it with a `tracks:` list; entries with a `prompt` are generated."
        )
    return yaml.safe_load(path.read_text()) or {}


def generate_cue(api_key: str, prompt: str, out_path: Path, length_ms: int) -> None:
    response = requests.post(
        API_URL,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"prompt": prompt, "music_length_ms": length_ms},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    if out_path.stat().st_size == 0:
        out_path.unlink()
        raise RuntimeError("API returned an empty body")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("category", help="music category under data/music/")
    ap.add_argument("--only", help="generate just this entry (by file stem)")
    ap.add_argument("--force", action="store_true", help="regenerate existing files")
    ap.add_argument("--length-ms", type=int, default=DEFAULT_LENGTH_MS)
    ap.add_argument("--dry-run", action="store_true", help="show the plan, generate nothing")
    args = ap.parse_args()

    category_dir = MUSIC_ROOT / args.category
    manifest = load_manifest(category_dir / "tracks.yaml")

    targets = []
    for entry in manifest.get("tracks") or []:
        prompt = str(entry.get("prompt", "")).strip()
        if not prompt:
            continue  # a sourced (downloaded) track, not a generated one
        out_path = category_dir / str(entry.get("file", ""))
        if args.only and out_path.stem != args.only:
            continue
        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path.name} — exists (use --force to regenerate)")
            continue
        targets.append((entry, out_path, prompt))

    if not targets:
        print("Nothing to generate.")
        return 0

    seconds = args.length_ms / 1000
    print(f"{len(targets)} cue(s) to generate at {seconds:.0f}s each:\n")
    for entry, out_path, prompt in targets:
        print(f"  {out_path.name}")
        print(f"    mood: {', '.join(entry.get('mood') or [])}")
        print(f"    {prompt[:100]}...")
    if args.dry_run:
        print("\n(dry run — nothing generated)")
        return 0

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY not set (see .env.example)")

    print()
    failures = 0
    for entry, out_path, prompt in targets:
        print(f"  [gen]  {out_path.name} ...", flush=True)
        try:
            generate_cue(api_key, prompt, out_path, args.length_ms)
            size_mb = out_path.stat().st_size / 1e6
            print(f"  [ok]   {out_path.name} ({size_mb:.1f} MB)")
        except (RuntimeError, requests.RequestException) as e:
            print(f"  [FAIL] {out_path.name} — {e}")
            failures += 1

    if failures:
        print(f"\n{failures} cue(s) failed.", file=sys.stderr)
        return 1
    print(f"\nDone. Tracks land in {category_dir.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
