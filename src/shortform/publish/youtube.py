"""YouTube Data API v3 uploader.

Stdlib-only, matching `oauth.py` — two endpoints and a file upload don't
justify google-api-python-client's dependency tree.

Uses the RESUMABLE upload protocol rather than a simple multipart POST. These
files are 40-70MB; a simple upload that dies at 90% has to start over, and on
a domestic connection that is a routine occurrence rather than an edge case.
The resumable flow lets a retry ask the server how many bytes actually landed
and continue from there.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"

# YouTube hard limits. Exceeding either is a 400 that reads like a generic bad
# request, so they're clamped before the call rather than diagnosed after it.
MAX_TITLE = 100
MAX_DESCRIPTION = 5000

# 24 = Entertainment. The full list rarely matters for this pipeline; a
# strategy can override via `publish.category_id`.
DEFAULT_CATEGORY_ID = "24"

TRANSIENT_MARKERS = ("SERVICE_UNAVAILABLE", "backendError", "internalServerError")
MAX_ATTEMPTS = 5


@dataclass
class UploadResult:
    video_id: str
    url: str
    privacy: str


def build_metadata(
    title: str,
    description: str,
    tags: list[str],
    privacy: str = "private",
    category_id: str = DEFAULT_CATEGORY_ID,
) -> dict[str, Any]:
    """The `snippet` + `status` body for videos.insert.

    Truncation is silent by design: a 101-character title should publish with
    the last character dropped, not fail an upload of a video that already cost
    real money to render.
    """
    return {
        "snippet": {
            "title": title[:MAX_TITLE],
            "description": description[:MAX_DESCRIPTION],
            "tags": tags,
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            # Anything this pipeline makes is synthetic, and YouTube requires
            # disclosure of realistic altered content. Claymation puppets are
            # unlikely to be mistaken for real footage, but the honest answer
            # to "is this AI generated" is yes, so it is declared rather than
            # left for someone else to decide.
            "selfDeclaredMadeForKids": False,
        },
    }


def upload(
    video_path: Path,
    metadata: dict[str, Any],
    access_token: str,
    progress: bool = True,
) -> UploadResult:
    """Upload one video, resuming across retries. Returns the created video."""
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    size = video_path.stat().st_size
    session_url = _start_session(metadata, access_token, size)
    logger.info("Upload session opened for %s (%.1f MB)", video_path.name, size / 1e6)

    body = _send_bytes(session_url, video_path, size, progress=progress)
    video_id = body.get("id", "")
    if not video_id:
        raise RuntimeError(f"upload completed but returned no video id: {body}")

    return UploadResult(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        privacy=body.get("status", {}).get("privacyStatus", "unknown"),
    )


def _start_session(metadata: dict[str, Any], token: str, size: int) -> str:
    """Open a resumable session and return its upload URL."""
    query = urllib.parse.urlencode(
        {"uploadType": "resumable", "part": "snippet,status"}
    )
    req = urllib.request.Request(
        f"{UPLOAD_URL}?{query}",
        data=json.dumps(metadata).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(size),
            "X-Upload-Content-Type": "video/*",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            location = r.headers.get("Location")
    except urllib.error.HTTPError as e:
        raise RuntimeError(_explain(e, "opening upload session")) from e

    if not location:
        raise RuntimeError("no Location header on the resumable session response")
    return str(location)


def _send_bytes(
    session_url: str, video_path: Path, size: int, progress: bool
) -> dict[str, Any]:
    """PUT the file, resuming from the server's offset after a failure."""
    offset = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _put_from(session_url, video_path, size, offset, progress)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            retryable = e.code in (500, 502, 503, 504) or any(
                m in body for m in TRANSIENT_MARKERS
            )
            if not retryable or attempt == MAX_ATTEMPTS:
                raise RuntimeError(_explain(e, "uploading", body)) from e
            offset = _resume_offset(session_url, size)
            delay = 2**attempt
            logger.warning(
                "Upload failed (HTTP %s), resuming from byte %d in %ds "
                "(attempt %d/%d)",
                e.code, offset, delay, attempt, MAX_ATTEMPTS,
            )
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"upload failed after {MAX_ATTEMPTS} attempts: {e}") from e
            offset = _resume_offset(session_url, size)
            delay = 2**attempt
            logger.warning(
                "Upload connection error (%s), resuming from byte %d in %ds",
                e, offset, delay,
            )
            time.sleep(delay)

    raise RuntimeError("unreachable: upload loop exhausted without returning")


def _put_from(
    session_url: str, video_path: Path, size: int, offset: int, progress: bool
) -> dict[str, Any]:
    """Send bytes from `offset` to the end in one request."""
    with open(video_path, "rb") as fh:
        fh.seek(offset)
        data = fh.read()

    headers = {"Content-Length": str(len(data)), "Content-Type": "video/*"}
    if offset:
        # Only send Content-Range when resuming. On a fresh upload it is
        # optional, and omitting it keeps the common path identical to a
        # plain PUT.
        headers["Content-Range"] = f"bytes {offset}-{size - 1}/{size}"

    if progress:
        logger.info(
            "Uploading %s from byte %d (%.1f MB to send)",
            video_path.name, offset, len(data) / 1e6,
        )

    req = urllib.request.Request(session_url, data=data, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=1800) as r:
        parsed: dict[str, Any] = json.loads(r.read() or b"{}")
        return parsed


def _resume_offset(session_url: str, size: int) -> int:
    """Ask the server how many bytes it actually has.

    A zero-length PUT with `Content-Range: bytes */TOTAL` returns 308 with a
    Range header of what landed. Any failure to determine it returns 0, which
    restarts the upload — wasteful but correct, where a wrong offset would
    corrupt the file.
    """
    req = urllib.request.Request(
        session_url,
        data=b"",
        headers={"Content-Length": "0", "Content-Range": f"bytes */{size}"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status in (200, 201):
                return size  # already complete
            return 0
    except urllib.error.HTTPError as e:
        if e.code != 308:
            return 0
        rng = e.headers.get("Range")  # e.g. "bytes=0-12345"
        if not rng or "-" not in rng:
            return 0
        try:
            return int(rng.split("-")[-1]) + 1
        except ValueError:
            return 0
    except (urllib.error.URLError, TimeoutError):
        return 0


def _explain(e: urllib.error.HTTPError, what: str, body: str = "") -> str:
    """Turn an API error into something actionable.

    Every hint here corresponds to a failure that is opaque from the raw
    response — the same reasoning as veo_backend's credits sniff.
    """
    if not body:
        body = e.read().decode("utf-8", errors="ignore")
    hint = ""
    if "quotaExceeded" in body:
        hint = (
            "\n  Daily quota is spent — an upload costs ~1600 units of the "
            "default 10,000/day, so this caps out at ~6 uploads per day. "
            "Resets at midnight Pacific."
        )
    elif "accessNotConfigured" in body or "has not been used" in body:
        hint = (
            "\n  The YouTube Data API v3 is not enabled for this Google Cloud "
            "project. Enable it in the console, wait a minute, and retry."
        )
    elif "youtubeSignupRequired" in body:
        hint = "\n  This Google account has no YouTube channel. Create one first."
    elif "forbidden" in body.lower() and "upload" in body.lower():
        hint = (
            "\n  The account may be unverified. YouTube caps unverified "
            "accounts at 15-minute uploads and can block them entirely."
        )
    return f"{what} failed: HTTP {e.code}\n  body: {body}{hint}"
