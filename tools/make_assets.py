#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Generate the app icon and splash screen into imagedit/assets/.

Run from the project root:  python tools/make_assets.py
Pure PIL, fully reproducible (fixed RNG seed).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "imagedit" / "assets"

# The icon mark: the character 像 ("image / portrait"), top half intact,
# bottom half pixelated, split horizontally.
ICON_CHAR = "像"

# Editable pixelation grid for the icon's bottom half (used by the icon
# designer). Cells are half-blocks (B2 = size/32) for finer detail. Cell
# (j, i) maps to canvas pixels: x = (GRID_X0 + j) * B2, y = (GRID_Y0 + i) * B2.
GRID_X0, GRID_Y0 = 2, 16       # grid origin in half-block units
GRID_COLS, GRID_ROWS = 28, 12  # full content width x bottom half
PATTERN_PATH = ASSETS / "icon_pattern.json"


def load_pattern() -> list[tuple[int, int]] | None:
    """Load a hand-designed keep-pattern (from the icon designer), if any."""
    if PATTERN_PATH.exists():
        data = json.loads(PATTERN_PATH.read_text())
        return [tuple(c) for c in data.get("keep", [])]
    return None

BG = (36, 31, 49)          # deep indigo
ACCENT = (0, 200, 255)     # cyan used in the UI overlays
SILHOUETTE = (232, 230, 240)

MOSAIC_TINTS = [
    (98, 90, 125), (130, 120, 160), (70, 64, 92),
    (160, 152, 190), (52, 47, 70), (112, 104, 142),
]


def _font(size: int, bold: bool = True):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _find_cjk_font(px: int) -> ImageFont.FreeTypeFont:
    """Locate a font containing CJK glyphs across macOS/Linux/Windows."""
    candidates = [
        os.environ.get("IMAGEDIT_CJK_FONT"),
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        # Linux
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/tmp/NotoSansSC-Bold.otf",
        # Windows
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, px)
            except Exception:
                continue
    raise RuntimeError(
        f"No CJK font found to render {ICON_CHAR!r}. Set IMAGEDIT_CJK_FONT "
        "to a font file containing Chinese glyphs.")


def _glyph_mask(size: int) -> Image.Image:
    """The icon character as an L-mode mask, scaled to span 2B..14B vertically
    (so the horizontal split at 8B is its exact midline) and centered."""
    B = size // 16
    target_h = 12 * B
    font = _find_cjk_font(target_h)
    pad = target_h
    canvas = Image.new("L", (3 * pad, 3 * pad), 0)
    ImageDraw.Draw(canvas).text((pad, pad), ICON_CHAR, font=font, fill=255)
    bbox = canvas.getbbox()
    if bbox is None:
        raise RuntimeError(f"Font rendered no pixels for {ICON_CHAR!r}.")
    glyph = canvas.crop(bbox)
    gw, gh = glyph.size
    new_w = min(14 * B, max(1, round(gw * target_h / gh)))
    glyph = glyph.resize((new_w, target_h), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    mask.paste(glyph, (size // 2 - new_w // 2, 2 * B))
    return mask


def silhouette_coverage(size: int = 512) -> dict[tuple[int, int], float]:
    """Per-cell glyph coverage for the editable grid (designer hints)."""
    from PIL import ImageStat
    B2 = size // 32
    mask = _glyph_mask(size)
    cov = {}
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            x0, y0 = (GRID_X0 + j) * B2, (GRID_Y0 + i) * B2
            cell = mask.crop((x0, y0, x0 + B2, y0 + B2))
            cov[(j, i)] = ImageStat.Stat(cell).mean[0] / 255.0
    return cov


def make_icon(size: int = 512,
              pattern: list[tuple[int, int]] | None = None) -> Image.Image:
    """The character 像 ("image"), split horizontally: original | redacted.

    Top half: intact smooth white glyph. Bottom half: binary pixelation on a
    half-block grid (B2 = size/32) - each cell is either kept white or erased
    to the background. Strictly two tones plus the accent split line.

    The glyph is scaled to span exactly 2B..14B vertically, so the horizontal
    split at 8B is its precise midline and the pixel grid is anchored to the
    glyph's top and bottom baselines.

    If `pattern` is given (or imagedit/assets/icon_pattern.json exists, saved
    from tools/icon_designer), those exact cells are kept instead of the
    random per-row selection - letting you encode initials or any motif.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    B = size // 16
    B2 = size // 32
    m = B  # background margin
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 6, fill=BG)

    mask = _glyph_mask(size)
    split_y = 8 * B

    # Top half: intact smooth glyph (the "original").
    top = Image.new("L", (size, size), 0)
    top.paste(mask.crop((0, 0, size, split_y)), (0, 0))
    img.paste(SILHOUETTE, (0, 0), top)

    # Bottom half ("redacted"): hand-designed pattern, or random pixelation.
    if pattern is None:
        pattern = load_pattern()
    if pattern is not None:
        for j, i in pattern:
            if not (0 <= j < GRID_COLS and 0 <= i < GRID_ROWS):
                continue
            x0, y0 = (GRID_X0 + j) * B2, (GRID_Y0 + i) * B2
            d.rectangle([x0, y0, x0 + B2 - 1, y0 + B2 - 1], fill=SILHOUETTE)
        return _finish_icon(d, img, B, m, size)

    from PIL import ImageStat
    rng = random.Random(7)
    keep_prob = 0.55
    for i in range(GRID_ROWS):
        y0 = (GRID_Y0 + i) * B2
        # Candidate cells of this row that lie on the glyph.
        row_cells = []
        for j in range(GRID_COLS):
            x0 = (GRID_X0 + j) * B2
            cell = mask.crop((x0, y0, x0 + B2, y0 + B2))
            if ImageStat.Stat(cell).mean[0] / 255.0 >= 0.5:
                row_cells.append(x0)
        if not row_cells:
            continue
        # Exact share per row (random positions) so density is even.
        n_keep = max(1, round(keep_prob * len(row_cells)))
        for x0 in rng.sample(row_cells, n_keep):
            d.rectangle([x0, y0, x0 + B2 - 1, y0 + B2 - 1], fill=SILHOUETTE)

    return _finish_icon(d, img, B, m, size)


def _finish_icon(d: ImageDraw.ImageDraw, img: Image.Image, B: int,
                 m: int, size: int) -> Image.Image:
    # Horizontal split line: original (top) | redacted (bottom).
    split_y = 8 * B
    lw = max(2, size // 170)
    d.rectangle([int(1.2 * B), split_y - lw // 2,
                 int(14.8 * B), split_y - lw // 2 + lw - 1], fill=ACCENT)
    # Thin accent border.
    d.rounded_rectangle([m, m, size - m, size - m], radius=size // 6,
                        outline=ACCENT, width=max(2, size // 128))
    return img


def make_splash(w: int = 640, h: int = 360) -> Image.Image:
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)

    # Mosaic band across the bottom.
    rng = random.Random(3)
    block = 24
    for by in range(h - 3 * block, h, block):
        for bx in range(0, w, block):
            d.rectangle([bx, by, bx + block - 1, by + block - 1],
                        fill=rng.choice(MOSAIC_TINTS))

    # Small icon motif on the left.
    icon = make_icon(160)
    img.paste(icon, (40, 70), icon)

    # Title + tagline.
    d.text((230, 95), "ImagEdit", font=_font(56), fill=(245, 245, 250))
    d.text((233, 170), "mosaic & blur redaction", font=_font(22, bold=False),
           fill=(170, 168, 185))
    try:
        import imagedit
        version = imagedit.__version__
    except Exception:
        version = "0.1.0"
    d.text((233, 205), f"v{version}", font=_font(16, bold=False),
           fill=(120, 118, 138))

    # Accent line.
    d.rectangle([230, 158, 470, 161], fill=ACCENT)
    return img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon = make_icon(512)
    icon.save(ASSETS / "icon.png")
    make_icon(128).save(ASSETS / "icon_128.png")
    # Multi-size Windows .ico (used by PyInstaller / the installer).
    icon.save(ASSETS / "icon.ico",
              sizes=[(16, 16), (32, 32), (48, 48), (64, 64),
                     (128, 128), (256, 256)])
    make_splash().save(ASSETS / "splash.png")
    src = "icon_pattern.json" if PATTERN_PATH.exists() else "random pattern"
    print(f"Wrote icon.png, icon_128.png, icon.ico, splash.png to {ASSETS} "
          f"({src})")


if __name__ == "__main__":
    main()
