"""Assembly stage — FFmpeg combines images + audio into final video."""

from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shortform.config import PROJECT_ROOT, MusicConfig
from shortform.models.script import Segment, WordTiming
from shortform.models.video import VideoStatus
from shortform.pipeline.context import PipelineContext
from shortform.store.file_store import FileStore
from shortform.visuals.backend import VisualOutputType

logger = logging.getLogger(__name__)


class AssemblyStage:
    @property
    def name(self) -> str:
        return "assembly"

    def validate(self, ctx: PipelineContext) -> list[str]:
        errors: list[str] = []
        segment_types = ctx.artifacts.get("segment_visual_types", {})
        segment_clip_lists: dict[int, list[str]] = ctx.artifacts.get("segment_clips", {})

        for seg in ctx.script.segments:
            if not seg.audio_path or not Path(seg.audio_path).exists():
                errors.append(f"Segment {seg.index}: missing audio at {seg.audio_path}")
            if not seg.image_path or not Path(seg.image_path).exists():
                errors.append(f"Segment {seg.index}: missing visual at {seg.image_path}")

            # Multi-clip VIDEO segments concat from segment_clips[seg.index]; the
            # sub-clips beyond clip 0 (== image_path) aren't covered by the check
            # above. On a resume against a cleaned asset dir a missing sub-clip
            # would otherwise pass validate() and fail deep in ffmpeg with an
            # opaque error.
            if segment_types.get(seg.index) == VisualOutputType.VIDEO:
                sub_clips = segment_clip_lists.get(seg.index, [])
                if len(sub_clips) > 1:
                    for ci, clip in enumerate(sub_clips):
                        p = Path(clip)
                        if not p.exists() or p.stat().st_size == 0:
                            errors.append(
                                f"Segment {seg.index}: missing/empty sub-clip {ci} at {clip}"
                            )

        # Check ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            errors.append("ffmpeg not found in PATH")

        return errors

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        file_store = FileStore()
        vid_cfg = ctx.settings.video
        vis_cfg = ctx.settings.visuals
        segments = ctx.script.segments

        work_dir = file_store.video_dir(ctx.video.id)
        output_path = file_store.final_video_path(ctx.video.id, ctx.script.title)

        # Per-segment visual types (supports mixed Veo/Pillow fallback)
        segment_types = ctx.artifacts.get("segment_visual_types", {})
        # Per-segment multi-clip paths (set by visual_gen when one Veo clip
        # wasn't enough to cover the F5-TTS narration duration).
        segment_clip_lists: dict[int, list[str]] = ctx.artifacts.get(
            "segment_clips", {}
        )

        # Step 1: Create per-segment video clips (visual + audio, no subtitles yet)
        segment_clips: list[Path] = []
        for seg in segments:
            clip_path = work_dir / f"clip_{seg.index:02d}.mp4"
            seg_type = segment_types.get(seg.index, VisualOutputType.IMAGE)

            if seg_type == VisualOutputType.VIDEO:
                # Multi-clip case: concat sub-clips into one segment-video first
                # so that the muxed clip's video length >= audio length.
                sub_clip_paths = segment_clip_lists.get(seg.index, [])
                if len(sub_clip_paths) > 1:
                    concat_path = work_dir / f"segment_{seg.index:02d}_concat.mp4"
                    _concat_video_clips_with_xfade(
                        clips=[Path(p) for p in sub_clip_paths],
                        output_path=concat_path,
                        crossfade_duration=vis_cfg.crossfade_duration,
                        fps=vid_cfg.fps,
                        width=vid_cfg.width,
                        height=vid_cfg.height,
                        video_bitrate=vid_cfg.video_bitrate,
                        pixel_format=vid_cfg.pixel_format,
                        preset=vid_cfg.preset,
                    )
                    video_input = concat_path
                else:
                    video_input = Path(seg.image_path)

                # Veo clip(s) → mux with audio
                _mux_video_with_audio(
                    video_path=video_input,
                    audio_path=Path(seg.audio_path),
                    output_path=clip_path,
                    video_bitrate=vid_cfg.video_bitrate,
                    pixel_format=vid_cfg.pixel_format,
                    preset=vid_cfg.preset,
                )
            else:
                # Still image (Pillow or Veo fallback) — apply Ken Burns zoom
                _create_segment_clip(
                    image_path=Path(seg.image_path),
                    audio_path=Path(seg.audio_path),
                    output_path=clip_path,
                    duration=seg.actual_duration,
                    zoom=vis_cfg.ken_burns_zoom,
                    width=vid_cfg.width,
                    height=vid_cfg.height,
                    fps=vid_cfg.fps,
                    video_bitrate=vid_cfg.video_bitrate,
                    pixel_format=vid_cfg.pixel_format,
                    preset=vid_cfg.preset,
                )

            # Step 1b: Burn animated subtitles onto the clip
            if seg.word_timings:
                subtitled_path = work_dir / f"clip_{seg.index:02d}_sub.mp4"
                _burn_animated_subtitles(
                    clip_path=clip_path,
                    output_path=subtitled_path,
                    segment=seg,
                    work_dir=work_dir,
                    width=vid_cfg.width,
                    height=vid_cfg.height,
                    fps=vid_cfg.fps,
                    font_size=vis_cfg.font_size,
                    font_color=vis_cfg.font_color,
                    video_bitrate=vid_cfg.video_bitrate,
                    pixel_format=vid_cfg.pixel_format,
                    preset=vid_cfg.preset,
                )
                clip_path = subtitled_path

            segment_clips.append(clip_path)
            logger.info("Created clip for segment %d (%.1fs)", seg.index, seg.actual_duration)

        # Step 2: Concatenate clips with crossfades
        assembled_path = output_path
        music_cfg = ctx.settings.music
        music_category = ctx.strategy.music.get("category", "")
        has_music = music_cfg.enabled and music_category
        if has_music:
            # Assemble to a temp file — music gets mixed in Step 3
            assembled_path = work_dir / "assembled_no_music.mp4"

        if len(segment_clips) == 1:
            _remux_with_faststart(segment_clips[0], assembled_path)
        else:
            _concat_with_crossfade(
                clips=segment_clips,
                output_path=assembled_path,
                crossfade_duration=vis_cfg.crossfade_duration,
                width=vid_cfg.width,
                height=vid_cfg.height,
                fps=vid_cfg.fps,
                video_bitrate=vid_cfg.video_bitrate,
                audio_bitrate=vid_cfg.audio_bitrate,
                audio_sample_rate=vid_cfg.audio_sample_rate,
                pixel_format=vid_cfg.pixel_format,
                preset=vid_cfg.preset,
            )

        # Step 3: Mix background music (if configured and tracks available)
        if has_music:
            music_track = _pick_music_track(music_cfg, music_category)
            if music_track:
                strategy_volume = ctx.strategy.music.get("volume")
                strategy_duck = ctx.strategy.music.get("duck_volume")
                music_volume = (
                    strategy_volume if strategy_volume is not None else music_cfg.volume
                )
                duck_volume = (
                    strategy_duck if strategy_duck is not None else music_cfg.duck_volume
                )
                _mix_background_music(
                    video_path=assembled_path,
                    music_path=music_track,
                    output_path=output_path,
                    music_volume=music_volume,
                    duck_volume=duck_volume,
                    fade_in=music_cfg.fade_in,
                    fade_out=music_cfg.fade_out,
                    audio_bitrate=vid_cfg.audio_bitrate,
                    audio_sample_rate=vid_cfg.audio_sample_rate,
                )
                logger.info("Mixed background music: %s", music_track.name)
            else:
                logger.warning("No music tracks found in category '%s'", music_category)
                if assembled_path != output_path:
                    assembled_path.rename(output_path)

        ctx.video.output_path = str(output_path)
        ctx.video.file_size_bytes = file_store.get_video_size(output_path)
        ctx.video.status = VideoStatus.ASSEMBLED

        logger.info(
            "Assembly complete: %s (%.1f MB)",
            output_path.name,
            ctx.video.file_size_bytes / (1024 * 1024),
        )
        return ctx


_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text based on actual rendered pixel width."""
    words = text.split()
    lines: list[str] = []
    current_line: list[str] = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def _group_words_into_phrases(
    word_timings: list[WordTiming],
    max_words: int = 3,
) -> list[tuple[str, float, float]]:
    """Group word timings into short display phrases.

    Returns list of (phrase_text, start_time, end_time).
    """
    phrases: list[tuple[str, float, float]] = []
    i = 0
    while i < len(word_timings):
        chunk = word_timings[i : i + max_words]
        text = " ".join(w.word for w in chunk)
        start = chunk[0].start
        # End time: start of next phrase, or end of last word in chunk
        if i + max_words < len(word_timings):
            end = word_timings[i + max_words].start
        else:
            last = chunk[-1]
            end = last.start + last.duration
        phrases.append((text, start, end))
        i += max_words
    return phrases


def _render_subtitle_frame(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_size: int,
    font_color: tuple[int, int, int],
    width: int,
    height: int,
) -> Image.Image:
    """Render a single subtitle phrase as a transparent PNG frame."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 80
    max_width = width - 2 * margin
    lines = _wrap_text_to_width(draw, text, font, max_width)

    line_height = font_size + 10
    total_text_height = line_height * len(lines)
    # Position in lower third of frame
    y_start = int(height * 0.70) - total_text_height // 2

    shadow_offset = 3
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = y_start + i * line_height
        # Dark background pill for readability
        pill_pad = 12
        draw.rounded_rectangle(
            [x - pill_pad, y - pill_pad // 2,
             x + text_width + pill_pad, y + font_size + pill_pad // 2],
            radius=10,
            fill=(0, 0, 0, 160),
        )
        draw.text((x + shadow_offset, y + shadow_offset), line, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=font, fill=(*font_color, 255))

    return img


def _burn_animated_subtitles(
    clip_path: Path,
    output_path: Path,
    segment: Segment,
    work_dir: Path,
    width: int,
    height: int,
    fps: int,
    font_size: int,
    font_color: str,
    video_bitrate: str,
    pixel_format: str,
    preset: str,
) -> None:
    """Overlay phrase-by-phrase animated subtitles onto a video clip.

    Renders each phrase as a transparent PNG and composites them using
    FFmpeg overlay filters with enable timing expressions.
    """
    phrases = _group_words_into_phrases(segment.word_timings)
    if not phrases:
        return

    # Load font
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont = ImageFont.load_default()
    for fp in _FONT_PATHS:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    h = font_color.lstrip("#")
    rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    sub_dir = work_dir / f"subs_{segment.index:02d}"
    sub_dir.mkdir(exist_ok=True)

    # Render a PNG for each phrase
    phrase_paths: list[Path] = []
    for idx, (text, _start, _end) in enumerate(phrases):
        frame = _render_subtitle_frame(text, font, font_size, rgb, width, height)
        frame_path = sub_dir / f"phrase_{idx:03d}.png"
        frame.save(str(frame_path), "PNG")
        phrase_paths.append(frame_path)

    # Build FFmpeg command: one input + overlay per phrase with enable timing
    inputs = ["-i", str(clip_path)]
    for p in phrase_paths:
        inputs.extend(["-i", str(p)])

    # Chain overlays: [0:v] → overlay phrase 0 → overlay phrase 1 → ...
    filter_parts: list[str] = []
    n = len(phrases)
    for idx, (_text, start, end) in enumerate(phrases):
        in_label = "[0:v]" if idx == 0 else f"[tmp{idx}]"
        out_label = "[vout]" if idx == n - 1 else f"[tmp{idx + 1}]"
        enable = f"between(t\\,{start:.3f}\\,{end:.3f})"
        filter_parts.append(
            f"{in_label}[{idx + 1}:v]overlay=0:0:enable='{enable}'{out_label}"
        )

    filter_complex = ";".join(filter_parts)

    _run_ffmpeg(
        inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-b:v", video_bitrate,
            "-pix_fmt", pixel_format,
            "-preset", preset,
            "-c:a", "copy",
            str(output_path),
        ]
    )


def _run_ffmpeg(args: list[str]) -> None:
    """Run an ffmpeg command, raising on failure."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")


def _mux_video_with_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    video_bitrate: str,
    pixel_format: str,
    preset: str,
) -> None:
    """Mux a pre-animated video clip with its audio track.

    Applies a short audio fade-out before the cutoff so the inter-segment
    crossfade doesn't inherit a hard audio boundary.

    The multi-clip path usually makes the video at least as long as the
    narration, but a degraded run (chained-clip generation failed twice,
    leaving a single ~8s Veo clip against 14-20s of F5-TTS) can leave the
    video shorter. We must NOT let -shortest truncate the audio in that case —
    that silently drops the back half of the narration. Instead we hold the
    video's last frame (tpad clone) out to the narration length so all speech
    is preserved.
    """
    video_duration = _probe_duration(video_path)
    audio_duration = _probe_duration(audio_path)
    fade_ms = 0.05  # 50ms micro-fade — imperceptible but prevents pops

    if video_duration + fade_ms < audio_duration:
        pad = audio_duration - video_duration
        logger.warning(
            "Segment video (%.1fs) shorter than narration (%.1fs); holding last "
            "frame for %.1fs so no speech is truncated",
            video_duration, audio_duration, pad,
        )
        output_duration = audio_duration
        video_filter = f"[0:v]tpad=stop_mode=clone:stop_duration={pad:.3f}[vout]"
    else:
        output_duration = min(video_duration, audio_duration)
        video_filter = "[0:v]copy[vout]"

    fade_start = max(0, output_duration - fade_ms)

    _run_ffmpeg([
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex",
        f"{video_filter};[1:a]afade=t=out:st={fade_start:.3f}:d={fade_ms}[aout]",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-b:v", video_bitrate,
        "-pix_fmt", pixel_format,
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        str(output_path),
    ])


def _create_segment_clip(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
    duration: float,
    zoom: float,
    width: int,
    height: int,
    fps: int,
    video_bitrate: str,
    pixel_format: str,
    preset: str,
) -> None:
    """Create a video clip from a still image + audio with Ken Burns zoom."""
    # Use ffprobe for authoritative audio duration, but for a still-image clip
    # a probe failure isn't fatal — fall back to the estimated duration so Ken
    # Burns still renders rather than aborting the whole assembly.
    try:
        audio_duration = _probe_duration(audio_path)
    except RuntimeError:
        logger.warning(
            "ffprobe failed on %s; using estimated duration %.1fs", audio_path, duration
        )
        audio_duration = duration
    if audio_duration <= 0:
        audio_duration = duration  # fallback to estimated

    # Generate extra frames so video is always longer than audio — -shortest trims to audio
    padded_frames = int(audio_duration * fps) + fps  # +1 second buffer

    # Ken Burns: slow zoom from 1.0 to zoom factor over the clip duration
    zoom_filter = (
        f"[0:v]scale={int(width * zoom)}:{int(height * zoom)},"
        f"zoompan=z='1+({zoom - 1})*on/{max(1, padded_frames)}':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={padded_frames}:s={width}x{height}:fps={fps}[vout]"
    )

    # Micro fade-out at end prevents pops at crossfade boundaries
    fade_ms = 0.05
    fade_start = max(0, audio_duration - fade_ms)
    audio_filter = f"[1:a]afade=t=out:st={fade_start:.3f}:d={fade_ms}[aout]"

    _run_ffmpeg([
        "-loop", "1",
        "-i", str(image_path),
        "-i", str(audio_path),
        "-filter_complex", f"{zoom_filter};{audio_filter}",
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-b:v", video_bitrate,
        "-pix_fmt", pixel_format,
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-shortest",
        str(output_path),
    ])


def _concat_with_crossfade(
    clips: list[Path],
    output_path: Path,
    crossfade_duration: float,
    width: int,
    height: int,
    fps: int,
    video_bitrate: str,
    audio_bitrate: str,
    audio_sample_rate: int,
    pixel_format: str,
    preset: str,
) -> None:
    """Concatenate clips with video crossfade and audio crossfade transitions.

    Defensive per-input normalization before xfade/acrossfade because the
    pre-muxed segment clips inherit whatever timebase / framerate / sample
    rate the upstream Veo + F5-TTS produced, and we've seen mismatches:
      - timebase (1/12288 vs 1/15360) → xfade parse error
      - framerate (24/1 vs 25/1) → xfade parse error
      - (defensive) audio sample rate inconsistencies → acrossfade weirdness

    Video chain per input: settb=AVTB,setpts=PTS-STARTPTS,fps=N,scale=W:H,format=PIXFMT
    Audio chain per input: asettb=AVTB,asetpts=PTS-STARTPTS,aresample=R
    """
    if len(clips) < 2:
        raise ValueError("Need at least 2 clips for crossfade concatenation")

    # Build filter_complex for xfade between consecutive clips
    inputs: list[str] = []
    for clip in clips:
        inputs.extend(["-i", str(clip)])

    # Build xfade chain
    n = len(clips)
    cf = crossfade_duration
    filter_parts: list[str] = []

    # Get durations for offset calculation
    durations = [_probe_duration(clip) for clip in clips]

    # Per-input normalization
    v_norm = (
        f"settb=AVTB,setpts=PTS-STARTPTS,fps={fps},"
        f"scale={width}:{height},format={pixel_format}"
    )
    a_norm = f"asettb=AVTB,asetpts=PTS-STARTPTS,aresample={audio_sample_rate}"
    for i in range(n):
        filter_parts.append(f"[{i}:v]{v_norm}[vn{i}]")
        filter_parts.append(f"[{i}:a]{a_norm}[an{i}]")

    # Video crossfade chain (over normalized streams)
    prev_label = "[vn0]"
    offset = 0.0
    for i in range(1, n):
        offset += durations[i - 1] - cf
        out_label = f"[v{i}]" if i < n - 1 else "[vout]"
        filter_parts.append(
            f"{prev_label}[vn{i}]xfade=transition=fade:duration={cf}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label

    # Audio crossfade chain (over normalized streams)
    prev_label = "[an0]"
    offset = 0.0
    for i in range(1, n):
        offset += durations[i - 1] - cf
        out_label = f"[a{i}]" if i < n - 1 else "[aout]"
        filter_parts.append(
            f"{prev_label}[an{i}]acrossfade=d={cf}:c1=tri:c2=tri{out_label}"
        )
        prev_label = out_label

    filter_complex = ";".join(filter_parts)

    _run_ffmpeg(
        inputs
        + [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-b:v", video_bitrate,
            "-pix_fmt", pixel_format,
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", audio_bitrate,
            "-ar", str(audio_sample_rate),
            "-movflags", "+faststart",
            str(output_path),
        ]
    )


def _concat_video_clips_with_xfade(
    clips: list[Path],
    output_path: Path,
    crossfade_duration: float,
    fps: int,
    width: int,
    height: int,
    video_bitrate: str,
    pixel_format: str,
    preset: str,
) -> None:
    """Concatenate sub-clips of one segment with video-only xfades.

    Used when a segment has >1 Veo clip because the F5-TTS narration runs
    longer than one Veo clip can cover. Output is video-only; audio gets
    muxed in the next step from the segment's F5-TTS WAV.

    Each input is normalized before the xfade chain because Veo's outputs
    are not 100% consistent across calls — we've seen timebase mismatches
    (1/12288 vs 1/15360) and framerate mismatches (24/1 vs 25/1) in the
    wild. xfade rejects either mismatch with a parse error. The
    normalization chain handles all known mismatches:
      - settb=AVTB,setpts=PTS-STARTPTS — uniform timebase (AV_TIME_BASE)
      - fps=N — uniform framerate via frame resampling
      - scale=W:H — uniform dimensions (no-op if already matching)
      - format=PIXFMT — uniform pixel format
    """
    if len(clips) < 2:
        raise ValueError("Need at least 2 clips for sub-clip xfade concat")

    inputs: list[str] = []
    for clip in clips:
        inputs.extend(["-i", str(clip)])

    n = len(clips)
    cf = crossfade_duration
    durations = [_probe_duration(c) for c in clips]

    # Normalize timebase, framerate, dimensions, and pixel format per input
    norm_chain = (
        f"settb=AVTB,setpts=PTS-STARTPTS,fps={fps},"
        f"scale={width}:{height},format={pixel_format}"
    )
    norm_parts = [f"[{i}:v]{norm_chain}[vn{i}]" for i in range(n)]

    # xfade chain over normalized inputs
    xfade_parts: list[str] = []
    prev_label = "[vn0]"
    offset = 0.0
    for i in range(1, n):
        offset += durations[i - 1] - cf
        out_label = f"[v{i}]" if i < n - 1 else "[vout]"
        xfade_parts.append(
            f"{prev_label}[vn{i}]xfade=transition=fade:duration={cf}:"
            f"offset={offset:.3f}{out_label}"
        )
        prev_label = out_label

    _run_ffmpeg(
        inputs + [
            "-filter_complex", ";".join(norm_parts + xfade_parts),
            "-map", "[vout]",
            "-an",
            "-c:v", "libx264",
            "-profile:v", "high",
            "-b:v", video_bitrate,
            "-pix_fmt", pixel_format,
            "-preset", preset,
            str(output_path),
        ]
    )


def _remux_with_faststart(input_path: Path, output_path: Path) -> None:
    """Copy a single clip and add faststart flag."""
    _run_ffmpeg([
        "-i", str(input_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ])


def _probe_duration(path: Path) -> float:
    """Get the duration of a media file using ffprobe.

    Raises RuntimeError on ffprobe failure or unparseable output rather than
    silently returning 0.0 — a zero duration poisons every downstream
    calculation (xfade offsets go negative, the -shortest mux cutoff becomes
    0, the music atrim length becomes 0), turning a genuinely broken/missing
    input into a silent-but-wrong render. Callers that have a meaningful
    fallback (e.g. an estimated duration for a still image) should catch this.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path} (exit {result.returncode}): "
            f"{result.stderr.strip()[:300]}"
        )
    out = result.stdout.strip()
    try:
        return float(out)
    except ValueError as e:
        raise RuntimeError(
            f"ffprobe returned an unparseable duration for {path}: {out!r}"
        ) from e


def _pick_music_track(music_cfg: MusicConfig, category: str) -> Path | None:
    """Pick a random music track from the category directory."""
    music_dir = PROJECT_ROOT / music_cfg.music_dir / category
    if not music_dir.exists():
        return None
    tracks = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
    return random.choice(tracks) if tracks else None


def _mix_background_music(
    video_path: Path,
    music_path: Path,
    output_path: Path,
    music_volume: float,
    duck_volume: float,
    fade_in: float,
    fade_out: float,
    audio_bitrate: str,
    audio_sample_rate: int,
) -> None:
    """Mix background music under the narration with volume ducking.

    Uses FFmpeg's sidechaincompress to duck the music when speech is present,
    with fade in/out at the video boundaries.
    """
    video_duration = _probe_duration(video_path)

    # Music filter chain:
    # 1. Loop music to cover video duration, trim to match
    # 2. Set base volume
    # 3. Fade in at start, fade out at end
    # 4. Sidechain compress: duck music when narration audio is present
    fade_out_start = max(0, video_duration - fade_out)

    # amix weights the already-ducked music relative to the narration. The
    # weight is duck_volume/music_volume because the music stream was already
    # scaled by music_volume upstream; this cancels back to duck_volume. Guard
    # against music_volume == 0 (a valid "mute the bed" config that is not None,
    # so it slips past the caller's None check) — otherwise this is a
    # ZeroDivisionError in the terminal assembly stage after all generation.
    duck_weight = duck_volume / music_volume if music_volume else 0.0

    # The narration is [0:a], music is [1:a]
    # sidechaincompress: music (source) is ducked by narration (sidechain)
    # threshold=0.02 = duck when narration exceeds ~-34dB (catches all speech)
    # ratio=5 = strong ducking
    # attack/release = how fast ducking kicks in/releases
    filter_complex = (
        f"[1:a]aloop=loop=-1:size=2e+09,atrim=duration={video_duration:.3f},"
        f"volume={music_volume},"
        f"afade=t=in:d={fade_in},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}"
        f"[music];"
        f"[music][0:a]sidechaincompress="
        f"threshold=0.02:ratio=5:attack=0.1:release=0.5:level_sc=1"
        f"[ducked];"
        f"[0:a][ducked]amix=inputs=2:duration=first:weights=1 {duck_weight:.2f}"
        f"[aout]"
    )

    _run_ffmpeg([
        "-i", str(video_path),
        "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(audio_sample_rate),
        "-movflags", "+faststart",
        str(output_path),
    ])
