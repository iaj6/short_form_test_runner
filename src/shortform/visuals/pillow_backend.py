"""Pillow backend — gradient backgrounds + text overlay (free, local, instant)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from shortform.models.script import Segment
from shortform.visuals.backend import VisualOutput, VisualOutputType

logger = logging.getLogger(__name__)

FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


class PillowBackend:
    @property
    def name(self) -> str:
        return "pillow"

    async def generate(
        self,
        segment: Segment,
        output_path: Path,
        width: int,
        height: int,
        config: dict[str, Any],
    ) -> VisualOutput:
        font_size = config.get("font_size", 64)
        font_color = config.get("font_color", "#FFFFFF")
        text_margin = config.get("text_margin", 80)
        gradient_top = config.get("gradient_top", "#1a1a2e")
        gradient_bottom = config.get("gradient_bottom", "#16213e")

        font = _load_font(font_size)

        img = _create_gradient_frame(
            width=width,
            height=height,
            color_top=gradient_top,
            color_bottom=gradient_bottom,
        )

        _draw_text_overlay(
            img=img,
            text=segment.text_overlay,
            font=font,
            color=font_color,
            margin=text_margin,
        )

        out = output_path.with_suffix(".png")
        img.save(str(out), "PNG")

        return VisualOutput(
            path=out,
            output_type=VisualOutputType.IMAGE,
            width=width,
            height=height,
        )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_PATHS:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    logger.warning("No TrueType font found, using default bitmap font")
    return ImageFont.load_default()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _create_gradient_frame(
    width: int, height: int, color_top: str, color_bottom: str
) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    r1, g1, b1 = _hex_to_rgb(color_top)
    r2, g2, b2 = _hex_to_rgb(color_bottom)

    for y in range(height):
        ratio = y / height
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


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


def _draw_text_overlay(
    img: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    color: str,
    margin: int,
) -> None:
    if not text:
        return

    draw = ImageDraw.Draw(img)
    width = img.width

    max_width = width - 2 * margin
    lines = _wrap_text_to_width(draw, text, font, max_width)

    line_height = font.size + 10
    total_height = line_height * len(lines)
    y_start = (img.height - total_height) // 2

    rgb = _hex_to_rgb(color)

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = y_start + i * line_height

        shadow_offset = 3
        draw.text(
            (x + shadow_offset, y + shadow_offset), line, font=font, fill=(0, 0, 0)
        )
        draw.text((x, y), line, font=font, fill=rgb)
