"""Find candidate juxtaposition moments in a YouTube political video.

Test-item v0 of the political-juxtaposition format ("Pod Save framing":
let the speaker run against themselves via clip selection alone). Given
a YouTube URL — typically a White House upload, press conference, or
political speech — this script:

1. Downloads the video + auto-generated English captions via yt-dlp
2. Builds a clean timestamped transcript from the VTT
3. Asks Claude to identify N standalone-clip-worthy moments where the
   speaker reveals out-of-touch priorities, trivial focus during serious
   times, self-aggrandizement, or contradictions worth juxtaposing
4. Cuts each candidate as an MP4 with ffmpeg

Output goes to data/moments/<video_id>/:
- video.mp4              cached source download
- captions.en.vtt        cached source captions
- moments.json           Claude's rationale + timestamps for the run
- moment_NN.mp4          extracted clip candidates

Usage:
    uv run python scripts/find_moments.py "https://www.youtube.com/watch?v=..."
    uv run python scripts/find_moments.py URL --count 8 --pad 2.0

This is exploratory infra — no pairing, no overlay, no publishing.
Eyeball the clips: if 1-2 of every 5 land, the format is worth pursuing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import yt_dlp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MOMENTS_DIR = PROJECT_ROOT / "data" / "moments"

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_COUNT = 5
DEFAULT_PAD_PRE = 1.5   # seconds of lead-in before the cited start
DEFAULT_PAD_POST = 1.0  # seconds of trail after the cited end

SYSTEM_PROMPT = """\
You curate clip candidates from political speech transcripts for a \
juxtaposition-format short-form video channel. The channel pairs a \
politician's own words against unrelated real-world footage — no \
narration, no overlay, no commentary. The whole effect comes from \
selection, so your job is to find the most self-incriminating moments.

What makes a strong candidate:
- Self-contained: the quote lands without needing setup or surrounding context
- Suggests a real-world counterpoint: a viewer can immediately picture what \
  footage would land hardest paired against it
- Tonally distinct: boastful, dismissive, weirdly fixated on something \
  trivial, contradicting a known reality, or out of step with current events
- Length: 5-25 seconds of speech, ideally a complete thought
- Specific: a vague platitude is weak; a specific number, claim, brag, or \
  fixation is strong

What to AVOID:
- Generic political boilerplate ("we love this country", "America is great")
- Moments that require audio cues you can't see in the transcript \
  (laughter, audience reaction)
- Repetitive call-and-response or list-recitation
- Moments where you'd need to fabricate a misleading counterpoint to make \
  them work — we want honest juxtaposition, not deceptive editing

QUOTE & TIMESTAMP RULES — read carefully:
- The quote MUST be a verbatim CONTIGUOUS span from the transcript. \
  No ellipses ("..."), no paraphrasing, no skipping over middle portions, \
  no re-ordering. Copy the words exactly as they appear in the transcript.
- start_seconds MUST equal the [MM:SS] timestamp of the FIRST transcript \
  line whose words begin your quote. Convert MM:SS to seconds (MM*60+SS).
- end_seconds MUST equal the [MM:SS] timestamp of the FIRST transcript line \
  AFTER the last line of your quote. If your quote ends at the last line of \
  the transcript, use that line's timestamp + 5.
- Do NOT use the timestamp of where "the punch lands" or the most striking \
  phrase — always the first line of the quote, even if there's brief lead-up.

Be ruthless. Five excellent candidates beats fifteen mediocre ones."""


def vtt_timestamp_to_seconds(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_mmss(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def parse_vtt(path: Path) -> list[tuple[float, str]]:
    """Parse a YouTube auto-caption VTT into deduped (start_seconds, text) cues.

    YT auto-captions are noisy: rolling 2-line context, per-word timing tags,
    duplicated lines across consecutive cues. We strip the inline tags, take
    one cue per text-line, and dedupe by exact-text-seen.
    """
    raw = path.read_text(encoding="utf-8")
    cues: list[tuple[float, str]] = []
    for block in re.split(r"\r?\n\r?\n", raw):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        timing_idx = next(
            (i for i, ln in enumerate(lines) if "-->" in ln),
            -1,
        )
        if timing_idx < 0:
            continue
        m = re.search(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->",
            lines[timing_idx],
        )
        if not m:
            continue
        start = vtt_timestamp_to_seconds(m.group(1))
        text = " ".join(lines[timing_idx + 1:])
        text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)
        text = re.sub(r"</?c[^>]*>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            cues.append((start, text))

    deduped: list[tuple[float, str]] = []
    seen: set[str] = set()
    for start, text in cues:
        if text in seen:
            continue
        seen.add(text)
        deduped.append((start, text))
    return deduped


def format_transcript_for_llm(cues: list[tuple[float, str]]) -> str:
    return "\n".join(f"[{seconds_to_mmss(s)}] {t}" for s, t in cues)


def download_video(url: str, dest_dir: Path) -> tuple[str, Path, Path]:
    """Download video + English auto-captions. Returns (video_id, video_path, vtt_path)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-orig"],
        "subtitlesformat": "vtt",
        "outtmpl": str(dest_dir / "video.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id: str = info["id"]  # type: ignore[index]
    video_path = next(
        (p for p in dest_dir.glob("video.*") if p.suffix in {".mp4", ".mkv", ".webm"}),
        None,
    )
    if video_path is None:
        raise RuntimeError(f"yt-dlp finished but no video file found in {dest_dir}")

    vtt_candidates = sorted(dest_dir.glob("video.*.vtt"))
    if not vtt_candidates:
        raise RuntimeError(
            f"No English captions available for {url}. v0 requires auto-captions; "
            "Whisper transcription is not yet wired up."
        )
    vtt_path = vtt_candidates[0]
    return video_id, video_path, vtt_path


# ----- Claude moment-finding ---------------------------------------------------

RECORD_MOMENTS_TOOL = {
    "name": "record_moments",
    "description": (
        "Record the chosen candidate moments from the transcript. "
        "Call this exactly once with all candidates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_seconds": {
                            "type": "number",
                            "description": "Start time of the moment in seconds.",
                        },
                        "end_seconds": {
                            "type": "number",
                            "description": (
                                "End time of the moment in seconds. "
                                "Should be 5-25 seconds after start."
                            ),
                        },
                        "quote": {
                            "type": "string",
                            "description": "The verbatim quote in this window.",
                        },
                        "why_interesting": {
                            "type": "string",
                            "description": (
                                "1-2 sentences on why this is a strong "
                                "juxtaposition candidate."
                            ),
                        },
                        "contrast_topic": {
                            "type": "string",
                            "description": (
                                "What real-world footage or news topic would "
                                "pair against this most effectively. Be specific."
                            ),
                        },
                    },
                    "required": [
                        "start_seconds",
                        "end_seconds",
                        "quote",
                        "why_interesting",
                        "contrast_topic",
                    ],
                },
            }
        },
        "required": ["moments"],
    },
}


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _word_prefix_overlap(a: list[str], b: list[str]) -> int:
    """Length of leading-word overlap between a and b."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _word_suffix_overlap(a: list[str], b: list[str]) -> int:
    """Length of trailing-word overlap between a and b."""
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def anchor_moment_to_cues(
    quote: str,
    cues: list[tuple[float, str]],
    min_overlap: int = 3,
    window: int = 60,
) -> tuple[float, float] | None:
    """Re-derive (start_seconds, end_seconds) by matching quote against cues.

    Claude sometimes picks start_seconds at "where the punch lands" rather than
    the first line of the quote. This function ignores Claude's claimed times
    and finds boundaries by best word-prefix overlap (for start) and best
    word-suffix overlap (for end), within a `window` of cues from the start.

    Returns None if no confident match — caller should fall back.
    """
    if not quote.strip() or not cues:
        return None

    quote_words = _normalize(quote).split()
    if len(quote_words) < 4:
        return None

    first_idx: int | None = None
    best_first = 0
    for i, (_, text) in enumerate(cues):
        cue_words = _normalize(text).split()
        if not cue_words:
            continue
        overlap = _word_prefix_overlap(cue_words, quote_words)
        if overlap >= min_overlap and overlap > best_first:
            best_first = overlap
            first_idx = i

    if first_idx is None:
        return None

    # End-cue search via accumulator: walk forward concatenating cue text;
    # the smallest j where the full quote is a substring of the accumulated
    # cues is our end. Robust to filler words ("yep?") trailing the cue.
    last_idx = first_idx
    upper = min(first_idx + window, len(cues))
    quote_norm = " ".join(quote_words)
    accumulated = ""
    found_end = False
    for j in range(first_idx, upper):
        cue_norm = " ".join(_normalize(cues[j][1]).split())
        if not cue_norm:
            continue
        accumulated = (accumulated + " " + cue_norm).strip() if accumulated else cue_norm
        if quote_norm in accumulated:
            last_idx = j
            found_end = True
            break

    # Accumulator fallback: if quote isn't found as substring (e.g. minor
    # paraphrase), use word-suffix-overlap to pick the best candidate.
    if not found_end:
        best_last = 0
        for j in range(first_idx, upper):
            cue_words = _normalize(cues[j][1]).split()
            if not cue_words:
                continue
            overlap = _word_suffix_overlap(cue_words, quote_words)
            if overlap >= min_overlap and overlap >= best_last:
                best_last = overlap
                last_idx = j

    start_s = cues[first_idx][0]
    if last_idx + 1 < len(cues):
        end_s = cues[last_idx + 1][0]
    else:
        end_s = cues[last_idx][0] + 4.0

    if end_s <= start_s:
        return None
    return start_s, end_s


def find_moments_with_claude(
    api_key: str,
    transcript: str,
    count: int,
    model: str,
) -> list[dict]:
    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f"Find the {count} strongest juxtaposition-clip candidates in the "
        f"transcript below. Timestamps are in [MM:SS] form at the start of "
        f"each line. Use those to derive start_seconds. End_seconds should be "
        f"the natural end of the chosen thought (5-25s after start).\n\n"
        f"Transcript:\n\n{transcript}"
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=[RECORD_MOMENTS_TOOL],
        tool_choice={"type": "tool", "name": "record_moments"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "record_moments":
            return list(block.input["moments"])  # type: ignore[index]
    raise RuntimeError(
        "Claude did not call record_moments. Raw response: "
        f"{response.model_dump_json()[:500]}"
    )


# ----- Clip extraction ---------------------------------------------------------


def cut_clip(
    src: Path,
    dest: Path,
    start: float,
    end: float,
    pad_pre: float,
    pad_post: float,
) -> None:
    """Cut a frame-accurate clip via ffmpeg re-encode."""
    real_start = max(0.0, start - pad_pre)
    duration = (end + pad_post) - real_start
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-ss", f"{real_start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting {dest.name}: {result.stderr}")


# ----- Entry point -------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="YouTube URL to mine for juxtaposition moments")
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Number of candidate moments to extract (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pad", type=float, default=None,
        help=(
            f"Apply same pad to pre and post (overrides defaults of "
            f"{DEFAULT_PAD_PRE}s pre, {DEFAULT_PAD_POST}s post)"
        ),
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="If video.mp4 + captions already exist for this URL, skip re-download",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env")
        sys.exit(1)

    pad_pre = args.pad if args.pad is not None else DEFAULT_PAD_PRE
    pad_post = args.pad if args.pad is not None else DEFAULT_PAD_POST

    # Pre-resolve video id so we can reuse cache
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(args.url, download=False)
    video_id: str = info["id"]  # type: ignore[index]
    title: str = info.get("title", "(untitled)")  # type: ignore[union-attr]

    out_dir = MOMENTS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Video: {title}")
    print(f"ID: {video_id}")
    print(f"Output: {out_dir.relative_to(PROJECT_ROOT)}/\n")

    cached_video = next(
        (p for p in out_dir.glob("video.*") if p.suffix in {".mp4", ".mkv", ".webm"}),
        None,
    )
    cached_vtt = next(iter(sorted(out_dir.glob("video.*.vtt"))), None)

    if args.reuse and cached_video and cached_vtt:
        print(f"Reusing cached download ({cached_video.name}, {cached_vtt.name})")
        video_path = cached_video
        vtt_path = cached_vtt
    else:
        print("Downloading video + captions via yt-dlp...")
        _, video_path, vtt_path = download_video(args.url, out_dir)

    print(f"\nParsing captions: {vtt_path.name}")
    cues = parse_vtt(vtt_path)
    print(f"  {len(cues)} unique cue lines, "
          f"~{cues[-1][0]:.0f}s of speech" if cues else "  (no cues)")
    if not cues:
        print("ERROR: parsed transcript is empty; check the VTT format.")
        sys.exit(1)

    transcript = format_transcript_for_llm(cues)

    print(f"\nAsking {args.model} for {args.count} candidate moment(s)...")
    moments = find_moments_with_claude(
        api_key=api_key,
        transcript=transcript,
        count=args.count,
        model=args.model,
    )
    print(f"  Claude returned {len(moments)} candidate(s)")

    print("\nAnchoring + cutting clips...")
    for i, m in enumerate(moments):
        clip_path = out_dir / f"moment_{i:02d}.mp4"
        claimed_start = float(m["start_seconds"])
        claimed_end = float(m["end_seconds"])

        anchored = anchor_moment_to_cues(m["quote"], cues)
        if anchored is not None:
            start, end = anchored
            m["claimed_start_seconds"] = claimed_start
            m["claimed_end_seconds"] = claimed_end
            m["anchored"] = True
            m["start_seconds"] = start
            m["end_seconds"] = end
            drift = abs(start - claimed_start)
            drift_note = f" (drift {drift:+.1f}s vs claimed)" if drift > 0.5 else ""
        else:
            start, end = claimed_start, claimed_end
            m["anchored"] = False
            drift_note = " (anchor FAILED, using Claude's times)"

        try:
            cut_clip(
                video_path, clip_path,
                start=start, end=end,
                pad_pre=pad_pre, pad_post=pad_post,
            )
            print(f"  [{i:02d}] {seconds_to_mmss(start)}-{seconds_to_mmss(end)}"
                  f"  →  {clip_path.name}{drift_note}")
            print(f"       quote: {m['quote'][:90]}{'...' if len(m['quote']) > 90 else ''}")
            print(f"       contrast: {m['contrast_topic']}")
        except RuntimeError as e:
            print(f"  [{i:02d}] FAILED: {e}")

    manifest = {
        "video_id": video_id,
        "title": title,
        "url": args.url,
        "model": args.model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pad_pre": pad_pre,
        "pad_post": pad_post,
        "moments": moments,
    }
    manifest_path = out_dir / "moments.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"\nReview clips in: {out_dir.relative_to(PROJECT_ROOT)}/")
    print("If 1-2 of these land, the format is worth developing further.")


if __name__ == "__main__":
    main()
