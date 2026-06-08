# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Redaction primitives: mosaic (pixelate) and gaussian blur.

All functions operate on RGB uint8 numpy arrays (H, W, 3) and apply the
effect only where a boolean mask is True. They never mutate the input;
a new array is returned.
"""
from __future__ import annotations

import cv2
import numpy as np


def _validate(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an (H, W, 3) RGB array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")


def _full_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.ones(image.shape[:2], dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("mask shape must match image height/width")
    return mask.astype(bool)


def mosaic_full(image: np.ndarray, block_size: int = 12) -> np.ndarray:
    """Return a fully pixelated copy of the image.

    block_size is the side length (in pixels) of each mosaic tile; larger
    values produce coarser censorship.
    """
    _validate(image)
    block_size = max(2, int(block_size))
    h, w = image.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def blur_full(image: np.ndarray, strength: int = 15) -> np.ndarray:
    """Return a fully gaussian-blurred copy of the image.

    strength roughly corresponds to the blur radius; it is converted to an
    odd kernel size internally.
    """
    _validate(image)
    k = max(1, int(strength))
    k = k if k % 2 == 1 else k + 1  # gaussian kernel must be odd
    sigma = max(1.0, k / 3.0)
    return cv2.GaussianBlur(image, (k, k), sigma)


def _composite(image: np.ndarray, effect: np.ndarray, mask: np.ndarray,
               feather: int = 0, expand: bool = False) -> np.ndarray:
    """Blend `effect` over `image` where mask is True, with optional feather.

    feather > 0:
      - expand=False: a symmetric soft edge (gaussian on the mask). The redacted
        region's boundary fades both inward and outward.
      - expand=True (used for fill): the mask is first dilated OUTWARD by the
        feather amount, then the outer boundary is softened. The original region
        stays fully covered, and the visible edge sits *outside* the true
        contour - so the subject's precise outline isn't revealed.
    """
    out = image.copy()
    if not feather or feather <= 0:
        out[mask] = effect[mask]
        return out

    k = feather if feather % 2 == 1 else feather + 1
    m = mask.astype(np.float32)
    if expand:
        # Grow the mask outward, then soften only the new outer band.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        alpha = cv2.GaussianBlur(dilated.astype(np.float32), (k, k), 0)
        alpha = np.maximum(alpha, m)  # keep the original region fully opaque
    else:
        alpha = cv2.GaussianBlur(m, (k, k), 0)
    alpha = alpha[..., None]
    return (image * (1 - alpha) + effect * alpha).astype(np.uint8)


def apply_mosaic(image: np.ndarray, mask: np.ndarray | None = None,
                 block_size: int = 12, feather: int = 0) -> np.ndarray:
    """Apply a mosaic effect to the masked region (or whole image)."""
    _validate(image)
    mask = _full_mask(image, mask)
    effect = mosaic_full(image, block_size)
    return _composite(image, effect, mask, feather)


def apply_blur(image: np.ndarray, mask: np.ndarray | None = None,
               strength: int = 15, feather: int = 0) -> np.ndarray:
    """Apply a gaussian blur to the masked region (or whole image)."""
    _validate(image)
    mask = _full_mask(image, mask)
    effect = blur_full(image, strength)
    return _composite(image, effect, mask, feather)


def region_mean_color(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """Mean color of the masked region (used by 'auto' fill to blend in)."""
    if mask.any():
        r, g, b = image[mask].mean(axis=0)
        return int(r), int(g), int(b)
    return (0, 0, 0)


def apply_fill(image: np.ndarray, mask: np.ndarray | None = None,
               color: tuple[int, int, int] | None = (0, 0, 0),
               feather: int = 0) -> np.ndarray:
    """Fill the masked region with a solid color (hard redaction).

    color=None means "auto": use the mean color of the region itself, which
    blends the patch into its surroundings much more smoothly than true black.
    """
    _validate(image)
    mask = _full_mask(image, mask)
    if color is None:
        color = region_mean_color(image, mask)
    effect = np.empty_like(image)
    effect[:] = np.array(color, dtype=np.uint8)
    # Fill feather expands outward so the exact contour isn't outlined.
    return _composite(image, effect, mask, feather, expand=True)


def apply_blackout(image: np.ndarray, mask: np.ndarray | None = None,
                   color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Backward-compatible alias for a solid black fill."""
    return apply_fill(image, mask, color=color)


def rect_to_mask(shape: tuple[int, int], x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Build a boolean mask for an axis-aligned rectangle."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    mask[y0:y1, x0:x1] = True
    return mask


def polygon_to_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    """Build a boolean mask from a polygon (list of (x, y) points)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def circle_to_mask(shape: tuple[int, int], cx: int, cy: int, radius: int) -> np.ndarray:
    """Build a boolean mask for a filled circle (used by the brush)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), max(1, int(radius)), 1, thickness=-1)
    return mask.astype(bool)


# Registry so the UI can iterate available redaction styles.
REDACTION_STYLES = ("mosaic", "blur", "fill")


def apply_style(image: np.ndarray, mask: np.ndarray, style: str,
                strength: int = 15, block_size: int = 12,
                feather: int = 0,
                fill_color: tuple[int, int, int] | None = (0, 0, 0)) -> np.ndarray:
    """Dispatch to the requested redaction style by name.

    fill_color applies to the "fill" style; None means auto (region mean).
    """
    if style == "mosaic":
        return apply_mosaic(image, mask, block_size=block_size, feather=feather)
    if style == "blur":
        return apply_blur(image, mask, strength=strength, feather=feather)
    if style in ("fill", "blackout"):
        return apply_fill(image, mask, color=fill_color, feather=feather)
    raise ValueError(f"unknown redaction style: {style!r}")
