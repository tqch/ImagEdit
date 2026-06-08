# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Non-destructive redaction layers.

Each redaction (an AI subject, a rectangle, a lasso, or a brush stroke) is a
`RedactionLayer`: a mask plus the style/parameters used to redact it. The
original image is never modified; `LayerStack.composite()` rebuilds the result
by painting each visible layer over the original on demand. This makes every
parameter (feather, blur strength, mosaic block size, fill color) live-editable
and every redaction individually removable.

Each layer's effect is computed from the *original* image (so layers are
independent and predictable) and cached, keyed by its parameters, so dragging
one layer's slider only recomputes that layer.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import redaction

_id_counter = itertools.count(1)


@dataclass
class RedactionLayer:
    mask: np.ndarray                       # boolean (H, W)
    style: str = "mosaic"                  # mosaic | blur | fill
    strength: int = 15                     # blur kernel
    block_size: int = 12                   # mosaic tile size
    feather: int = 0                       # soft-edge radius
    fill_color: tuple[int, int, int] | None = (0, 0, 0)  # None = auto (mean)
    visible: bool = True
    name: str = "Layer"
    source: str = "manual"                 # ai | face | rect | lasso | brush
    id: int = field(default_factory=lambda: next(_id_counter))

    # Cached full-image effect + the key it was computed for.
    _effect: np.ndarray | None = field(default=None, repr=False, compare=False)
    _key: tuple | None = field(default=None, repr=False, compare=False)

    # ----- parameter snapshot (for undo, shares the mask reference) ---------
    def state(self) -> dict:
        return dict(style=self.style, strength=self.strength,
                    block_size=self.block_size, feather=self.feather,
                    fill_color=self.fill_color, visible=self.visible,
                    name=self.name, source=self.source, id=self.id)

    @classmethod
    def from_state(cls, mask: np.ndarray, st: dict) -> "RedactionLayer":
        layer = cls(mask=mask, style=st["style"], strength=st["strength"],
                    block_size=st["block_size"], feather=st["feather"],
                    fill_color=st["fill_color"], visible=st["visible"],
                    name=st["name"], source=st["source"])
        layer.id = st["id"]
        return layer

    def invalidate(self) -> None:
        self._effect = None
        self._key = None

    def _cache_key(self) -> tuple:
        return (self.style, self.strength, self.block_size,
                None if self.fill_color is None else tuple(self.fill_color))

    def effect(self, original: np.ndarray) -> np.ndarray:
        """Full-image redaction effect for this layer (cached)."""
        key = self._cache_key()
        if self._effect is None or self._key != key:
            if self.style == "mosaic":
                eff = redaction.mosaic_full(original, self.block_size)
            elif self.style == "blur":
                eff = redaction.blur_full(original, self.strength)
            else:  # fill
                color = self.fill_color
                if color is None:
                    color = redaction.region_mean_color(original, self.mask)
                eff = np.empty_like(original)
                eff[:] = np.array(color, dtype=np.uint8)
            self._effect = eff
            self._key = key
        return self._effect

    def render_onto(self, base: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Paint this layer's redaction over `base` (the running composite)."""
        if not self.visible or not self.mask.any():
            return base
        effect = self.effect(original)
        # Fill grows outward to hide the precise contour; mosaic/blur stay
        # symmetric.
        return redaction._composite(base, effect, self.mask, self.feather,
                                    expand=(self.style == "fill"))


class LayerStack:
    def __init__(self, original: np.ndarray):
        self.original = np.ascontiguousarray(original.astype(np.uint8))
        self.layers: list[RedactionLayer] = []

    def set_original(self, original: np.ndarray) -> None:
        self.original = np.ascontiguousarray(original.astype(np.uint8))
        self.layers.clear()

    def add(self, layer: RedactionLayer) -> RedactionLayer:
        self.layers.append(layer)
        return layer

    def remove(self, layer_id: int) -> None:
        self.layers = [ly for ly in self.layers if ly.id != layer_id]

    def get(self, layer_id: int) -> RedactionLayer | None:
        for ly in self.layers:
            if ly.id == layer_id:
                return ly
        return None

    def composite(self) -> np.ndarray:
        out = self.original.copy()
        for layer in self.layers:
            out = layer.render_onto(out, self.original)
        return out

    # ----- snapshot / restore for undo (masks shared by reference) ----------
    def snapshot(self) -> list[tuple[np.ndarray, dict]]:
        return [(ly.mask, ly.state()) for ly in self.layers]

    def restore(self, snap: list[tuple[np.ndarray, dict]]) -> None:
        self.layers = [RedactionLayer.from_state(mask, st) for mask, st in snap]
