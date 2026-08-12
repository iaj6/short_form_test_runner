"""Vision critic — checks a generated clip against its reference frame.

The pipeline could generate and assemble without ever looking at what it made.
Every failure this session was caught by a human pulling frames out of an MP4 and
eyeballing them: a character silently replaced by a different woman in a
different dress, a camera drifting until someone left the shot, clay texture
degrading three chain-hops deep. Unattended, none of that surfaces until you
watch forty finished episodes.

This module automates that check. It samples frames from a generated clip, sends
them to Claude alongside the reference image the clip was supposed to be
anchored to, and asks a narrow question: is this still the same scene with the
same characters?

DESIGN NOTES

*Severity, not a boolean.* Identity failures (wrong character, missing
character) are worth paying to regenerate. Mild generational softening is not —
it is inherent to chained generation and flagging it would burn credits on
clips that are perfectly usable. Only `fatal` issues trigger a regenerate.

*Biased toward passing.* A false positive costs one extra clip; a false negative
costs a broken episode. That asymmetry argues for strictness — but a critic that
flags everything is worse than none, because it burns the budget and trains you
to ignore it. So the prompt is explicit: flag clear, obvious failures; when
uncertain, pass.

*Never blocks the pipeline.* Any error — network, auth, refusal, malformed
response — degrades to "passed, unverified" with a warning. Soft-deps on the
same `anthropic` package the script/variant stages already require. A critic
that can crash a render is worse than no critic.
"""

from __future__ import annotations

import base64
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Opus 5. This runs once per generated clip, and a vision call is a rounding
# error against the Veo clip it might save — accuracy is worth more than the
# per-call cost here.
DEFAULT_MODEL = "claude-opus-5"

# Thinking is ON BY DEFAULT on Opus 5, and max_tokens caps thinking AND the tool
# call together — a tight limit truncates the verdict mid-reasoning and looks
# like a malformed response. Unused tokens aren't billed, so this is generous on
# purpose.
MAX_TOKENS = 8192

# Frames sampled per clip. The last frame matters most: it becomes the chain
# anchor for the next clip, so a failure there propagates forward.
DEFAULT_SAMPLE_COUNT = 3

FATAL = "fatal"
MINOR = "minor"

SYSTEM_PROMPT = """\
You are a continuity supervisor for an animated series. You compare frames from \
a newly generated shot against the reference image that shot was supposed to \
match, and report continuity failures.

You are checking for CONTINUITY, not quality. The generated frames come from a \
video model and will never be pixel-identical to the reference — lighting \
shifts, poses change, characters move and speak, the camera framing may differ \
slightly. None of that is a failure. Motion and expression changes are the \
entire point of the shot.

Report ONLY these, as `fatal`:
- A character in the reference has been REPLACED by a visibly different \
character (different face, different hair, different clothing).
- A character present in the reference is MISSING from the frames entirely, or \
has drifted so far out of frame they are effectively gone.
- Characters have SWAPPED positions relative to the reference (the one on the \
left is now on the right).
- A NEW character appears who is not in the reference.
- The setting has changed to a visibly different location.
- The frames are BLANK — solid black, solid white, or otherwise empty of any \
subject and setting. Report this as `blank_frames`.

Report these as `minor` (informational; they will not trigger a regenerate):
- Texture, sharpness, or color degradation relative to the reference.
- Set dressing that appeared or vanished (props, background details).
- Noticeable camera movement when the shot was meant to be locked off.

Judgment rules, in order of importance:
1. When uncertain, PASS. A costly regeneration of an acceptable shot is worse \
than letting a marginal one through.
2. Costume and identity are what you are really tracking. Judge them by \
distinctive, stable features — silhouette, hair, garment colour and shape — not \
by fine detail that a video model will naturally vary.
3. A character turning, gesturing, opening their mouth, or being partially \
occluded is NORMAL. Do not report it.
4. If you are given the shot's INTENDED ACTION, judge against it as well as \
against the reference. The reference is a single frozen frame from the start of \
the shot; the action is what the script asks to happen during it. When the \
action says a character leaves, exits, goes out, or is sent away, their absence \
later in the shot is the shot working — do NOT report `character_missing`. \
Likewise, a door, window, or piece of set the action requires is not a \
`new_character` or an unexplained change. A character the action does not \
mention should still be present throughout.
5. Rule 4 never excuses an empty shot. "The character exited" explains one \
character leaving frame; it does not explain black or empty frames, a vanished \
setting, or every character disappearing at once. Report those as `blank_frames` \
or `setting_changed` no matter what the intended action says.
"""

VERDICT_TOOL: dict[str, Any] = {
    "name": "report_continuity",
    "description": "Report continuity failures found in the generated frames.",
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "description": (
                    "Continuity failures found. Empty if the shot is acceptable."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "character_replaced",
                                "character_missing",
                                "characters_swapped",
                                "new_character",
                                "setting_changed",
                                "blank_frames",
                                "quality_degraded",
                                "set_dressing_changed",
                                "camera_moved",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": [FATAL, MINOR],
                            "description": (
                                "fatal = identity/staging broken, regenerate. "
                                "minor = informational only."
                            ),
                        },
                        "detail": {
                            "type": "string",
                            "description": "One sentence naming what changed.",
                        },
                    },
                    "required": ["kind", "severity", "detail"],
                },
            },
            "summary": {
                "type": "string",
                "description": "One short sentence describing the shot overall.",
            },
        },
        "required": ["issues", "summary"],
    },
}


@dataclass
class Issue:
    kind: str
    severity: str
    detail: str

    @property
    def is_fatal(self) -> bool:
        return self.severity == FATAL


@dataclass
class Verdict:
    """Outcome of reviewing one clip."""

    passed: bool
    issues: list[Issue] = field(default_factory=list)
    summary: str = ""
    # True when the check couldn't run (no key, API error, refusal). Such a
    # verdict always `passed` — but the caller should report it separately so an
    # unattended batch can't silently degrade into no verification at all.
    unverified: bool = False

    @property
    def fatal_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.is_fatal]

    def describe(self) -> str:
        if self.unverified:
            return f"unverified ({self.summary})"
        if not self.issues:
            return "ok"
        return "; ".join(f"[{i.severity}] {i.kind}: {i.detail}" for i in self.issues)


def extract_frames(
    video_path: Path, output_dir: Path, count: int = DEFAULT_SAMPLE_COUNT
) -> list[Path]:
    """Sample `count` frames spread across the clip.

    Always includes the last frame — it becomes the chain anchor for the
    following clip, so a failure there propagates into everything after it.
    """
    duration = _probe_duration(video_path)
    if duration <= 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    # Land just inside both ends rather than exactly on them; the true final
    # frame is often a compression artifact and seeking to 0.0 can miss.
    if count == 1:
        offsets = [duration * 0.5]
    else:
        span = duration - 0.3
        offsets = [0.15 + span * i / (count - 1) for i in range(count)]

    frames: list[Path] = []
    for i, offset in enumerate(offsets):
        out = output_dir / f"{video_path.stem}_critic_{i:02d}.png"
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{offset:.2f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "3", str(out),
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 0:
            frames.append(out)
    return frames


class ClipCritic:
    """Reviews generated clips against their reference frame."""

    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_count = sample_count

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def review(
        self,
        clip_path: Path,
        reference_path: Path,
        work_dir: Path,
        expected_characters: str = "",
        intended_action: str = "",
    ) -> Verdict:
        """Compare `clip_path` against `reference_path`.

        `intended_action` is what the script directs to happen during the shot.
        Without it the critic judges purely against a frozen reference frame, so
        a scripted exit is indistinguishable from the model losing a character —
        see `Segment.staged_action`.

        Returns a passing, `unverified` verdict on any failure to check — a
        broken critic must never block a render.
        """
        if not self.available:
            return Verdict(True, summary="no ANTHROPIC_API_KEY", unverified=True)
        if not reference_path.exists():
            return Verdict(
                True, summary=f"reference missing: {reference_path.name}", unverified=True
            )

        frames = extract_frames(clip_path, work_dir, self.sample_count)
        if not frames:
            return Verdict(
                True, summary=f"could not sample frames from {clip_path.name}",
                unverified=True,
            )

        try:
            return self._ask(
                frames, reference_path, expected_characters, intended_action
            )
        except Exception as e:  # noqa: BLE001 — a critic must not break a render
            logger.warning("Clip critic failed on %s: %s", clip_path.name, e)
            return Verdict(True, summary=f"critic error: {e}", unverified=True)

    def _ask(
        self,
        frames: list[Path],
        reference_path: Path,
        expected_characters: str,
        intended_action: str = "",
    ) -> Verdict:
        import anthropic

        content: list[dict[str, Any]] = [
            {"type": "text", "text": "REFERENCE IMAGE (the shot should match this):"},
            _image_block(reference_path),
            {
                "type": "text",
                "text": (
                    f"\nGENERATED FRAMES, in order across the shot "
                    f"({len(frames)} sampled):"
                ),
            },
        ]
        for frame in frames:
            content.append(_image_block(frame))

        ask = "Compare the generated frames against the reference and report continuity failures."
        if expected_characters:
            ask += f"\n\nCharacters who should be present: {expected_characters}."
        if intended_action:
            ask += (
                f"\n\nINTENDED ACTION for this shot: {intended_action}."
                "\nThe reference is the shot's starting frame, not its ending one."
                " Anything this action calls for is expected, not a failure."
            )
        content.append({"type": "text", "text": ask})

        client = anthropic.Anthropic(api_key=self.api_key)
        # The ignore below is for the SDK's strict overloads, which don't accept
        # the untyped dict payloads built above (same pattern as variant_select).
        response = client.messages.create(  # type: ignore[call-overload]
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "report_continuity"},
            messages=[{"role": "user", "content": content}],
        )

        # Opus 5 runs safety classifiers; a decline returns HTTP 200 with
        # stop_reason "refusal" and no usable content, so check before reading.
        if response.stop_reason == "refusal":
            return Verdict(True, summary="critic declined to review", unverified=True)

        for block in response.content:
            if block.type == "tool_use" and block.name == "report_continuity":
                return _verdict_from(dict(block.input))

        return Verdict(True, summary="critic returned no verdict", unverified=True)


def _verdict_from(payload: dict[str, Any]) -> Verdict:
    issues = [
        Issue(
            kind=str(raw.get("kind", "unknown")),
            severity=str(raw.get("severity", MINOR)),
            detail=str(raw.get("detail", "")),
        )
        for raw in payload.get("issues") or []
    ]
    return Verdict(
        passed=not any(i.is_fatal for i in issues),
        issues=issues,
        summary=str(payload.get("summary", "")),
    )


# Magic bytes -> media type. A file's extension is not evidence of its format:
# reference images written straight from an image API's response bytes can carry
# a .png name while actually being JPEG, and the API rejects the mismatch with a
# 400 that reads like a code bug rather than a data one.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def detect_media_type(data: bytes) -> str:
    """Media type from the bytes themselves, defaulting to PNG.

    WEBP needs a two-part check: "RIFF" then "WEBP" four bytes later.
    """
    for signature, media_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _image_block(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": detect_media_type(data),
            "data": base64.standard_b64encode(data).decode("utf-8"),
        },
    }


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0
