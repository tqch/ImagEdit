# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Text watermark / copyright overlay.

Draws a credit or copyright line over the (already redacted) image. Rendered
last in the composite and included in the saved file. Non-destructive: the
watermark is a single configurable object on the canvas, not baked into pixels
until export.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left",
             "center", "tiled")


# Fonts that include CJK (Chinese/Japanese/Korean) glyphs. Tried first when the
# watermark text contains CJK characters, so they render instead of tofu boxes.
# These also cover Latin, so they are a safe fallback generally.
_CJK_FONTS = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    # Windows
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/tmp/NotoSansSC-Bold.otf",
]
_LATIN_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _has_cjk(text: str) -> bool:
    return any(
        "⺀" <= ch <= "鿿"      # CJK radicals through unified ideographs
        or "぀" <= ch <= "ヿ"   # Hiragana / Katakana
        or "가" <= ch <= "힣"   # Hangul
        or "＀" <= ch <= "￯"   # fullwidth forms
        for ch in text)


def _find_font(px: int, text: str = "") -> ImageFont.FreeTypeFont:
    # If the text needs CJK glyphs, try CJK fonts first; otherwise Latin first.
    override = os.environ.get("IMAGEDIT_WATERMARK_FONT")
    if _has_cjk(text):
        order = [override] + _CJK_FONTS + _LATIN_FONTS
    else:
        order = [override] + _LATIN_FONTS + _CJK_FONTS
    for p in order:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                continue
    return ImageFont.load_default()


@dataclass
class Watermark:
    text: str = "© Your Name"
    position: str = "bottom-right"
    opacity: int = 50              # 0-100
    size_pct: float = 4.0          # font height as % of image height
    color: tuple[int, int, int] = (255, 255, 255)
    enabled: bool = False

    def apply(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a copy of the image with the watermark drawn on top."""
        if not self.enabled or not self.text.strip():
            return image_rgb
        h, w = image_rgb.shape[:2]
        base = Image.fromarray(image_rgb).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        px = max(10, int(h * self.size_pct / 100.0))
        font = _find_font(px, self.text)
        a = int(max(0, min(100, self.opacity)) / 100.0 * 255)
        fill = (*self.color, a)

        bbox = draw.textbbox((0, 0), self.text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin = max(6, px // 2)

        if self.position == "tiled":
            self._draw_tiled(draw, font, fill, tw, th, w, h)
        else:
            x, y = self._anchor(self.position, tw, th, w, h, margin, bbox)
            # Subtle shadow for legibility on busy backgrounds.
            draw.text((x + 2, y + 2), self.text, font=font,
                      fill=(0, 0, 0, int(a * 0.6)))
            draw.text((x, y), self.text, font=font, fill=fill)

        out = Image.alpha_composite(base, overlay).convert("RGB")
        return np.asarray(out, dtype=np.uint8)

    @staticmethod
    def _anchor(pos, tw, th, w, h, margin, bbox):
        ox, oy = bbox[0], bbox[1]  # font bearing offsets
        if pos == "bottom-right":
            x, y = w - tw - margin, h - th - margin
        elif pos == "bottom-left":
            x, y = margin, h - th - margin
        elif pos == "top-right":
            x, y = w - tw - margin, margin
        elif pos == "top-left":
            x, y = margin, margin
        else:  # center
            x, y = (w - tw) // 2, (h - th) // 2
        return x - ox, y - oy

    def _draw_tiled(self, draw, font, fill, tw, th, w, h):
        step_x = tw + max(40, tw // 2)
        step_y = th + max(40, th * 2)
        y = -th
        row = 0
        while y < h:
            offset = (step_x // 2) if row % 2 else 0
            x = -tw + offset
            while x < w:
                draw.text((x, y), self.text, font=font, fill=fill)
                x += step_x
            y += step_y
            row += 1
