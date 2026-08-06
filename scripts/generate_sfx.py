"""Generate sound-effect cues declared in data/sfx/library.yaml via ElevenLabs.

    uv run python scripts/generate_sfx.py
    uv run python scripts/generate_sfx.py --dry-run
    uv run python scripts/generate_sfx.py --only door_slam --force

Manifest-driven, like scripts/generate_music.py: any entry carrying a `prompt`
and lacking its audio file gets generated. The manifest is committed and the
audio is not, so the library is reproducible from the repo.

Generate a small REUSABLE library, not an effect per occurrence. A door slam is
a door slam — one file referenced by name from every scene that needs one is
cheaper and stays consistent across a long series.

Requires ELEVENLABS_API_KEY. Effects are short, so this is cheap; check current
pricing before generating a large library.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFX_DIR = PROJECT_ROOT / "data" / "sfx"
API_URL = "https://api.elevenlabs.io/v1/sound-generation"

DEFAULT_DURATION_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 180


def generate_effect(
    api_key: str, prompt: str, out_path: Path, duration_seconds: float
) -> None:
    response = requests.post(
        API_URL,
        headers={"xi-api-key": api_key, "Content-Type": "application/json"},
        json={"text": prompt, "duration_seconds": duration_seconds},
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
    ap.add_argument("--only", help="generate just this cue (by name)")
    ap.add_argument("--force", action="store_true", help="regenerate existing files")
    ap.add_argument("--dry-run", action="store_true", help="show the plan, generate nothing")
    args = ap.parse_args()

    manifest_path = SFX_DIR / "library.yaml"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text()) or {}

    targets = []
    for entry in manifest.get("effects") or []:
        prompt = str(entry.get("prompt", "")).strip()
        name = str(entry.get("name", ""))
        if not prompt or not name:
            continue
        out_path = SFX_DIR / str(entry.get("file", ""))
        if args.only and name != args.only:
            continue
        if out_path.exists() and not args.force:
            print(f"  [skip] {name} — exists (use --force to regenerate)")
            continue
        targets.append((entry, name, out_path, prompt))

    if not targets:
        print("Nothing to generate.")
        return 0

    print(f"{len(targets)} effect(s) to generate:\n")
    for entry, name, out_path, prompt in targets:
        seconds = float(entry.get("duration_seconds", DEFAULT_DURATION_SECONDS))
        print(f"  {name} ({seconds:.0f}s) — {entry.get('description', '')}")
    if args.dry_run:
        print("\n(dry run — nothing generated)")
        return 0

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY not set (see .env.example)")

    print()
    failures = 0
    for entry, name, out_path, prompt in targets:
        seconds = float(entry.get("duration_seconds", DEFAULT_DURATION_SECONDS))
        print(f"  [gen]  {name} ...", flush=True)
        try:
            generate_effect(api_key, prompt, out_path, seconds)
            print(f"  [ok]   {name} ({out_path.stat().st_size / 1000:.0f} KB)")
        except (RuntimeError, requests.RequestException) as e:
            print(f"  [FAIL] {name} — {e}")
            failures += 1

    if failures:
        print(f"\n{failures} effect(s) failed.", file=sys.stderr)
        return 1
    print(f"\nDone. Effects land in {SFX_DIR.relative_to(PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
