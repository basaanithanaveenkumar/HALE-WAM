"""Frame composition for the visualisation tools: fonts, headers, panels, badges and dividers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


# ---------------------------------------------------------------------------
# Lazy imports — avoid hard crash if libs aren't installed
# ---------------------------------------------------------------------------
def _import_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        from hale_core.exceptions import MissingDependencyError

        raise MissingDependencyError("cv2", "pip install opencv-python") from exc


def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _import_pil():
    from PIL import Image, ImageDraw, ImageFont
    return Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Font loading — DejaVu Sans is on every standard Linux distribution and
# supports Latin + extended characters, so curly-quotes, em-dashes, arrows
# etc. render properly instead of as ??? boxes.
# ---------------------------------------------------------------------------
_FONT_CACHE: Dict[Tuple[str, int], "object"] = {}


_FONT_CANDIDATES_REGULAR = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


_FONT_CANDIDATES_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


def get_font(size: int, bold: bool = False) -> Any:
    """Load a TTF font with a sensible fallback. Cached per (style, size)."""
    _, _, ImageFont = _import_pil()
    key = ("bold" if bold else "regular", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for path in candidates:
        if Path(path).is_file():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[key] = font
                return font
            except OSError as exc:  # unreadable / corrupt font file: try the next candidate
                logger.debug("skipping font {}: {}", path, exc)
    # Last resort: PIL's built-in bitmap font (won't be pretty but won't crash).
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def wrap_text(text: str, max_chars: int = 60) -> List[str]:
    """Wrap ``text`` to lines of at most ``max_chars`` characters (an empty text gives ``[""]``)."""
    return textwrap.wrap(text, width=max_chars) or [""]


# ---------------------------------------------------------------------------
# PIL drawing helpers — these replace the OpenCV Hershey-font text path,
# which can only render ASCII (anything else, including curly quotes and
# em-dashes that often appear in dataset prompts, shows as "???").
# ---------------------------------------------------------------------------
def _bgr_to_pil(frame_bgr: np.ndarray) -> PILImage:
    """Convert an OpenCV BGR uint8 frame to a PIL Image (RGB)."""
    Image, _, _ = _import_pil()
    return Image.fromarray(frame_bgr[:, :, ::-1].copy())


def _pil_to_bgr(img: PILImage) -> np.ndarray:
    """Convert a PIL Image (RGB) back to a BGR uint8 numpy frame."""
    arr = np.asarray(img.convert("RGB"))
    return arr[:, :, ::-1].copy()


def draw_header(img: PILImage, phase: str, sub: str = "",
                accent: Tuple[int, int, int] = (96, 200, 255)) -> None:
    """Draw a slim gradient header strip with phase label and an optional sub-label."""
    Image, ImageDraw, _ = _import_pil()
    W, H = img.size
    bar_h = 38

    # Gradient strip (dark teal -> accent)
    gradient = Image.new("RGB", (W, bar_h), (0, 0, 0))
    for x in range(W):
        t = x / max(W - 1, 1)
        r = int((1 - t) * 18 + t * accent[0] * 0.35)
        g = int((1 - t) * 22 + t * accent[1] * 0.35)
        b = int((1 - t) * 36 + t * accent[2] * 0.45)
        for y in range(bar_h):
            gradient.putpixel((x, y), (r, g, b))
    img.paste(gradient, (0, 0))

    draw = ImageDraw.Draw(img, "RGBA")
    # Accent left ribbon
    draw.rectangle((0, 0, 6, bar_h), fill=accent + (255,))
    # Bottom hairline divider
    draw.line((0, bar_h - 1, W, bar_h - 1), fill=(255, 255, 255, 70), width=1)

    title_font = get_font(20, bold=True)
    sub_font = get_font(14)
    draw.text((18, 8), phase, font=title_font, fill=(255, 255, 255))
    if sub:
        # Right-aligned sub label
        right_pad = 16
        bbox = draw.textbbox((0, 0), sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        draw.text((W - tw - right_pad, 11), sub, font=sub_font,
                  fill=(220, 230, 245))
    return bar_h


# Ordered phase labels and their accent colours for the PRAI bar.
_PRAI_PHASES = [
    ("01", "PERCEIVE", (96, 200, 255)),    # cyan
    ("02", "REASON",   (160, 120, 255)),   # purple
    ("03", "ACT",      (255, 180, 80)),    # orange
    ("04", "IMAGINE",  (130, 235, 170)),   # green
]


_PRAI_BAR_H = 32  # pixel height of the PRAI bar


def draw_prai_bar(img: PILImage, active_phase: int = 0) -> None:
    """
    Draw the Perceive → Reason → Act → Imagine progress bar directly
    below the header strip.  ``active_phase`` is 0-based (0=PERCEIVE …
    3=IMAGINE); that segment is visually highlighted.
    """
    Image, ImageDraw, _ = _import_pil()
    W, H = img.size
    bar_y = 38  # starts immediately below the 38 px header
    bar_h = _PRAI_BAR_H

    draw = ImageDraw.Draw(img, "RGBA")
    # Full-width background
    draw.rectangle((0, bar_y, W, bar_y + bar_h), fill=(10, 13, 25, 245))

    n = len(_PRAI_PHASES)
    seg_w = W // n

    num_font   = get_font(9,  bold=True)
    label_font = get_font(12, bold=True)

    for i, (num, label, accent) in enumerate(_PRAI_PHASES):
        x0 = i * seg_w
        x1 = (i + 1) * seg_w if i < n - 1 else W
        is_active = (i == active_phase)

        if is_active:
            # Subtle tinted fill
            tint = Image.new("RGBA", (x1 - x0, bar_h), (0, 0, 0, 0))
            td = ImageDraw.Draw(tint)
            td.rectangle(
                (0, 0, x1 - x0, bar_h),
                fill=(accent[0] // 5, accent[1] // 5, accent[2] // 5, 210),
            )
            # Left accent stripe
            td.rectangle((0, 0, 3, bar_h), fill=accent + (255,))
            # Bottom accent line
            td.rectangle((0, bar_h - 2, x1 - x0, bar_h), fill=accent + (200,))
            img.paste(tint, (x0, bar_y), tint)

            text_col = accent + (255,)
            num_col  = (accent[0], accent[1], accent[2], 170)
        else:
            text_col = (75, 85, 110, 255)
            num_col  = (50, 60, 80, 255)

        # Small "01" number in top-left of segment
        draw.text((x0 + 8, bar_y + 3), num, font=num_font, fill=num_col)

        # Centred phase label
        bb  = draw.textbbox((0, 0), label, font=label_font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        cx = x0 + (x1 - x0) // 2 - tw // 2
        cy = bar_y + (bar_h - th) // 2

        if is_active:
            draw.text((cx + 1, cy + 1), label, font=label_font,
                      fill=(0, 0, 0, 160))
        draw.text((cx, cy), label, font=label_font, fill=text_col)

        # Chevron separator
        if i < n - 1:
            cx2 = x1 - 1
            mid = bar_y + bar_h // 2
            draw.polygon(
                [(cx2 - 5, bar_y + 5), (cx2 + 2, mid), (cx2 - 5, bar_y + bar_h - 5)],
                fill=(40, 55, 80, 240),
            )

    # Bottom divider hairline
    draw.line((0, bar_y + bar_h - 1, W, bar_y + bar_h - 1),
              fill=(35, 50, 75, 200), width=1)


def draw_panel(img: PILImage, lines: List[Tuple[str, Tuple[int, int, int], bool]],
               anchor: str = "bottom", padding: int = 14, line_gap: int = 6,
               font_size: int = 17, label_font_size: int = 15,
               max_chars: int = 70, min_top: int = 44) -> None:
    """
    Draw a translucent rounded panel containing labelled text lines.

    ``lines`` is a list of (text, (r,g,b), is_label) tuples. Label lines render
    in bold and slightly smaller; body lines wrap to ``max_chars``.
    """
    Image, ImageDraw, _ = _import_pil()
    W, H = img.size

    body_font = get_font(font_size)
    label_font = get_font(label_font_size, bold=True)

    # Resolve wrapped text rows up-front so we can size the panel.
    rendered_rows: List[Tuple[str, Tuple[int, int, int], object, int]] = []
    for text, colour, is_label in lines:
        font = label_font if is_label else body_font
        for chunk in wrap_text(text, max_chars=max_chars):
            bbox = body_font.getbbox(chunk if chunk else " ")
            row_h = max(bbox[3] - bbox[1], font_size) + line_gap
            rendered_rows.append((chunk, colour, font, row_h))

    panel_h = sum(r[3] for r in rendered_rows) + padding * 2

    if anchor == "bottom":
        y0 = H - panel_h - 10
    else:
        y0 = 10
    y0 = max(y0, min_top)  # leave space for header + PRAI bar

    # Translucent rounded background
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    radius = 14
    margin = 12
    rect = (margin, y0, W - margin, y0 + panel_h)
    try:
        od.rounded_rectangle(rect, radius=radius, fill=(8, 10, 22, 200))
    except AttributeError:
        od.rectangle(rect, fill=(8, 10, 22, 200))
    # Subtle accent stripe on the left side of the panel
    od.rectangle((margin, y0, margin + 4, y0 + panel_h), fill=(96, 200, 255, 230))

    img.paste(overlay, (0, 0), overlay)

    draw = ImageDraw.Draw(img, "RGBA")
    y = y0 + padding
    for text, colour, font, row_h in rendered_rows:
        # 1px shadow for readability against busy backgrounds.
        draw.text((margin + 16, y + 1), text, font=font, fill=(0, 0, 0, 200))
        draw.text((margin + 15, y), text, font=font, fill=colour + (255,))
        y += row_h


def draw_correctness_badge(img: PILImage, is_correct: bool, top: int) -> None:
    """Small pill badge in the top-right area indicating CORRECT/WRONG."""
    Image, ImageDraw, _ = _import_pil()
    W, _ = img.size
    label = "CORRECT" if is_correct else "WRONG"
    colour = (60, 200, 120) if is_correct else (235, 90, 110)
    font = get_font(13, bold=True)

    draw = ImageDraw.Draw(img, "RGBA")
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 10, 4
    x1 = W - 16
    x0 = x1 - tw - pad_x * 2
    y0 = top + 2
    y1 = y0 + th + pad_y * 2
    try:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=10,
                               fill=colour + (220,))
    except AttributeError:
        draw.rectangle((x0, y0, x1, y1), fill=colour + (220,))
    draw.text((x0 + pad_x, y0 + pad_y - 1), label, font=font,
              fill=(255, 255, 255, 255))


def compose_frame(
    bgr_frame: np.ndarray,
    phase_label: str,
    sub_label: str,
    panel_lines: List[Tuple[str, Tuple[int, int, int], bool]],
    is_correct: Optional[bool] = None,
    accent: Tuple[int, int, int] = (96, 200, 255),
    active_phase: int = 0,
) -> np.ndarray:
    """Render header + PRAI bar + bottom info panel onto an image frame. Returns BGR uint8."""
    img = _bgr_to_pil(bgr_frame)
    header_h = draw_header(img, phase_label, sub_label, accent=accent)
    draw_prai_bar(img, active_phase=active_phase)
    top_offset = header_h + _PRAI_BAR_H
    if is_correct is not None:
        draw_correctness_badge(img, is_correct, top=top_offset + 4)
    draw_panel(img, panel_lines, anchor="bottom", min_top=top_offset + 6)
    return _pil_to_bgr(img)


def add_divider(top_img: np.ndarray, bottom_img: np.ndarray,
                divider_height: int = 6) -> np.ndarray:
    """Stack two BGR images with a thin gradient divider in between."""
    _, w, _ = top_img.shape
    divider = np.zeros((divider_height, w, 3), dtype=np.uint8)
    # Subtle teal accent strip.
    for y in range(divider_height):
        t = abs(y - divider_height / 2) / max(divider_height / 2, 1)
        intensity = int(180 * (1 - t))
        divider[y, :, 0] = max(intensity - 40, 12)
        divider[y, :, 1] = max(intensity - 20, 18)
        divider[y, :, 2] = max(intensity, 36)
    return np.vstack([top_img, divider, bottom_img])
