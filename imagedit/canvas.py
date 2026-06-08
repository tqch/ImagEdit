# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Interactive image canvas with non-destructive redaction layers.

The canvas keeps the pristine original image and a `LayerStack`. Every
redaction - AI subjects, rectangles, lassos, brush strokes - is added as a
layer; the displayed image is recomposited from the original on demand. This
makes each redaction's parameters live-editable and each layer individually
removable. Undo/redo snapshots the (lightweight) layer list.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon
from PyQt6.QtWidgets import QWidget

from . import redaction
from .layers import LayerStack, RedactionLayer
from .watermark import Watermark


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    """Convert an RGB uint8 numpy array to a QImage (copy, contiguous)."""
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class Canvas(QWidget):
    image_modified = pyqtSignal()
    status_message = pyqtSignal(str)
    history_changed = pyqtSignal()
    layers_changed = pyqtSignal()       # structure: add/remove/visibility/select

    MAX_HISTORY = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._stack: LayerStack | None = None
        self._composited: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self.zoom: float = 1.0

        self.selected_id: int | None = None
        self._active_brush_id: int | None = None
        self.watermark = Watermark()

        # AI subject visualization (colored masks + labelled boxes).
        self._subjects_overlay: QPixmap | None = None
        self._subject_boxes: list = []
        self._highlight_idx: int | None = None
        self.show_overlay: bool = True

        self.tool: str = "rect"  # rect | lasso | brush

        # Template settings for the NEXT redaction (the main window keeps these
        # in sync with the tool sliders).
        self.style: str = "mosaic"
        self.strength: int = 15
        self.block_size: int = 12
        self.brush_radius: int = 25
        self.feather: int = 0
        self.fill_color: tuple[int, int, int] = (0, 0, 0)
        self.fill_auto: bool = False

        # Undo/redo: stacks of LayerStack snapshots.
        self._undo: list = []
        self._redo: list = []

        # Transient interaction state.
        self._dragging = False
        self._rect_start: QPoint | None = None
        self._rect_end: QPoint | None = None
        self._lasso_pts: list[tuple[int, int]] = []
        self._cursor_pos: QPoint | None = None

    # ----- image management -------------------------------------------------
    def has_image(self) -> bool:
        return self._stack is not None

    def set_image(self, arr: np.ndarray, reset_history: bool = True) -> None:
        self._stack = LayerStack(arr)
        if reset_history:
            self._undo.clear()
            self._redo.clear()
            self.clear_subjects()
        self.selected_id = None
        self._active_brush_id = None
        self.render()
        self._apply_zoom_size()
        self.history_changed.emit()
        self.layers_changed.emit()

    def get_image(self) -> np.ndarray | None:
        """The currently composited (redacted) image - used for save / AI."""
        return None if self._composited is None else self._composited.copy()

    def get_original(self) -> np.ndarray | None:
        return None if self._stack is None else self._stack.original.copy()

    def image_shape(self) -> tuple[int, int]:
        if self._stack is None:
            return (0, 0)
        return self._stack.original.shape[:2]

    def render(self) -> None:
        """Recomposite original + visible layers and refresh the display."""
        if self._stack is None:
            self._composited = None
            self._pixmap = None
        else:
            comp = self._stack.composite()
            if self.watermark is not None and self.watermark.enabled:
                comp = self.watermark.apply(comp)
            self._composited = comp
            self._pixmap = QPixmap.fromImage(numpy_to_qimage(self._composited))
        self.update()
        self.image_modified.emit()

    # ----- layer access -----------------------------------------------------
    def layers(self) -> list[RedactionLayer]:
        return [] if self._stack is None else list(self._stack.layers)

    def get_layer(self, layer_id: int) -> RedactionLayer | None:
        return None if self._stack is None else self._stack.get(layer_id)

    def selected_layer(self) -> RedactionLayer | None:
        if self.selected_id is None:
            return None
        return self.get_layer(self.selected_id)

    def select_layer(self, layer_id: int | None) -> None:
        self.selected_id = layer_id
        # Selecting a non-brush layer ends the current brush session.
        if layer_id != self._active_brush_id:
            self._active_brush_id = None
        self.layers_changed.emit()
        self.update()

    def _new_name(self, source: str) -> str:
        n = sum(1 for ly in self.layers() if ly.source == source) + 1
        label = {"rect": "Rectangle", "lasso": "Lasso", "brush": "Brush",
                 "ai": "Person", "face": "Face"}.get(source, "Layer")
        return f"{label} {n}"

    def _template_layer(self, mask: np.ndarray, source: str,
                        style: str | None = None,
                        name: str | None = None) -> RedactionLayer:
        return RedactionLayer(
            mask=mask, style=style or self.style, strength=self.strength,
            block_size=self.block_size, feather=self.feather,
            fill_color=None if self.fill_auto else self.fill_color,
            source=source, name=name or self._new_name(source))

    def add_layer(self, mask: np.ndarray, source: str = "manual",
                  style: str | None = None, name: str | None = None,
                  select: bool = True) -> RedactionLayer | None:
        """Create a redaction layer from a mask using current template settings."""
        if self._stack is None or not mask.any():
            return None
        self._push_undo()
        layer = self._template_layer(mask, source, style=style, name=name)
        self._stack.add(layer)
        if select:
            self.selected_id = layer.id
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()
        return layer

    def update_layer(self, layer_id: int, **params) -> None:
        """Live-edit a layer's parameters and re-render (no undo push here)."""
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        for k, v in params.items():
            setattr(layer, k, v)
        layer.invalidate()
        self.render()

    def remove_layer(self, layer_id: int) -> None:
        if self._stack is None:
            return
        self._push_undo()
        self._stack.remove(layer_id)
        if self.selected_id == layer_id:
            self.selected_id = None
        if self._active_brush_id == layer_id:
            self._active_brush_id = None
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()

    def set_layer_visible(self, layer_id: int, visible: bool) -> None:
        layer = self.get_layer(layer_id)
        if layer is None:
            return
        self._push_undo()
        layer.visible = visible
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()

    def clear_layers(self) -> None:
        if self._stack is None or not self._stack.layers:
            return
        self._push_undo()
        self._stack.layers.clear()
        self.selected_id = None
        self._active_brush_id = None
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()

    # ----- history ----------------------------------------------------------
    def begin_edit(self) -> None:
        """Snapshot before a continuous edit gesture (e.g. a slider drag)."""
        self._push_undo()

    def _push_undo(self) -> None:
        if self._stack is None:
            return
        self._undo.append(self._stack.snapshot())
        if len(self._undo) > self.MAX_HISTORY:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def can_reset(self) -> bool:
        return self._stack is not None and bool(self._stack.layers)

    def undo(self) -> None:
        if not self._undo or self._stack is None:
            return
        self._redo.append(self._stack.snapshot())
        self._stack.restore(self._undo.pop())
        self._sync_selection()
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()
        self.status_message.emit("Undo")

    def redo(self) -> None:
        if not self._redo or self._stack is None:
            return
        self._undo.append(self._stack.snapshot())
        self._stack.restore(self._redo.pop())
        self._sync_selection()
        self.render()
        self.history_changed.emit()
        self.layers_changed.emit()
        self.status_message.emit("Redo")

    def reset_to_original(self) -> None:
        """Remove all layers, restoring the original image."""
        self.clear_layers()
        self.status_message.emit("Reset to original image")

    def _sync_selection(self) -> None:
        ids = {ly.id for ly in self.layers()}
        if self.selected_id not in ids:
            self.selected_id = None
        if self._active_brush_id not in ids:
            self._active_brush_id = None

    # ----- AI subject overlay ------------------------------------------------
    def set_subjects(self, subjects, colors, alpha: int = 100) -> None:
        if self._stack is None or not subjects:
            self.clear_subjects()
            return
        h, w = self.image_shape()
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        self._subject_boxes = []
        for s, c in zip(subjects, colors):
            overlay[s.mask] = (*c, alpha)
            self._subject_boxes.append((s.box, c, s.index))
        qimg = QImage(overlay.data, w, h, 4 * w,
                      QImage.Format.Format_RGBA8888).copy()
        self._subjects_overlay = QPixmap.fromImage(qimg)
        self.show_overlay = True
        self.update()

    def clear_subjects(self) -> None:
        self._subjects_overlay = None
        self._subject_boxes = []
        self._highlight_idx = None
        self.update()

    def set_overlay_visible(self, visible: bool) -> None:
        self.show_overlay = bool(visible)
        self.update()

    def highlight_subject(self, index: int | None) -> None:
        self._highlight_idx = index
        self.update()

    # ----- zoom -------------------------------------------------------------
    def set_zoom(self, zoom: float) -> None:
        self.zoom = max(0.05, min(8.0, zoom))
        self._apply_zoom_size()
        self.update()

    def _apply_zoom_size(self) -> None:
        if self._stack is None:
            return
        h, w = self.image_shape()
        self.setFixedSize(int(w * self.zoom), int(h * self.zoom))

    def wheelEvent(self, event):
        if self._stack is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.set_zoom(self.zoom * factor)
            event.accept()
        else:
            super().wheelEvent(event)

    # ----- coordinate mapping ----------------------------------------------
    def _to_image_coords(self, pos: QPoint) -> tuple[int, int]:
        x = int(pos.x() / self.zoom)
        y = int(pos.y() / self.zoom)
        h, w = self.image_shape()
        return max(0, min(w - 1, x)), max(0, min(h - 1, y))

    # ----- mouse handling ---------------------------------------------------
    def mousePressEvent(self, event):
        if self._stack is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        ix, iy = self._to_image_coords(event.position().toPoint())
        if self.tool == "rect":
            self._rect_start = QPoint(ix, iy)
            self._rect_end = QPoint(ix, iy)
        elif self.tool == "lasso":
            self._lasso_pts = [(ix, iy)]
        elif self.tool == "brush":
            self._brush_press(ix, iy)
        self.update()

    def mouseMoveEvent(self, event):
        self._cursor_pos = event.position().toPoint()
        if self._stack is None:
            self.update()
            return
        ix, iy = self._to_image_coords(event.position().toPoint())
        self.status_message.emit(f"({ix}, {iy})  zoom {self.zoom:.0%}")
        if self._dragging:
            if self.tool == "rect":
                self._rect_end = QPoint(ix, iy)
            elif self.tool == "lasso":
                self._lasso_pts.append((ix, iy))
            elif self.tool == "brush":
                self._brush_stamp(ix, iy)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._stack is None or not self._dragging:
            return
        self._dragging = False
        shape = self.image_shape()
        if self.tool == "rect" and self._rect_start and self._rect_end:
            mask = redaction.rect_to_mask(
                shape, self._rect_start.x(), self._rect_start.y(),
                self._rect_end.x(), self._rect_end.y())
            self.add_layer(mask, source="rect")
            self._rect_start = self._rect_end = None
        elif self.tool == "lasso" and len(self._lasso_pts) >= 3:
            mask = redaction.polygon_to_mask(shape, self._lasso_pts)
            self.add_layer(mask, source="lasso")
            self._lasso_pts = []
        self.update()

    # ----- brush ------------------------------------------------------------
    def _brush_press(self, ix: int, iy: int) -> None:
        """Start or continue a brush layer; strokes accumulate into one layer."""
        shape = self.image_shape()
        stamp = redaction.circle_to_mask(shape, ix, iy, self.brush_radius)
        layer = self.get_layer(self._active_brush_id) if self._active_brush_id else None
        if layer is None:
            # Start a new brush layer (undoable as one unit).
            self._push_undo()
            layer = self._template_layer(stamp, "brush")
            self._stack.add(layer)
            self._active_brush_id = layer.id
            self.selected_id = layer.id
            self.history_changed.emit()
            self.layers_changed.emit()
        else:
            # Append this stroke to the active brush layer.
            self._push_undo()
            layer.mask = layer.mask | stamp
            layer.invalidate()
        self.render()

    def _brush_stamp(self, ix: int, iy: int) -> None:
        layer = self.get_layer(self._active_brush_id) if self._active_brush_id else None
        if layer is None:
            return
        stamp = redaction.circle_to_mask(self.image_shape(), ix, iy, self.brush_radius)
        layer.mask = layer.mask | stamp
        layer.invalidate()
        self.render()

    def new_brush_session(self) -> None:
        """Next brush stroke starts a fresh layer instead of appending."""
        self._active_brush_id = None

    # ----- watermark --------------------------------------------------------
    def update_watermark(self, **params) -> None:
        for k, v in params.items():
            setattr(self.watermark, k, v)
        self.render()

    # ----- painting ---------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        if self._pixmap is None:
            painter.fillRect(self.rect(), QColor(40, 40, 40))
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Open an image to begin (File - Open)")
            return

        painter.drawPixmap(self.rect(), self._pixmap)

        # AI subject overlay: colored silhouettes + labelled boxes.
        if self._subjects_overlay is not None and self.show_overlay:
            painter.drawPixmap(self.rect(), self._subjects_overlay)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for box, color, idx in self._subject_boxes:
                x0, y0, x1, y1 = box
                rx0, ry0 = int(x0 * self.zoom), int(y0 * self.zoom)
                rw, rh = int((x1 - x0) * self.zoom), int((y1 - y0) * self.zoom)
                pen = QPen(QColor(*color), 4 if self._highlight_idx == idx else 2)
                painter.setPen(pen)
                painter.drawRect(rx0, ry0, rw, rh)
                tag = f" {idx} "
                fm = painter.fontMetrics()
                tw, th = fm.horizontalAdvance(tag), fm.height()
                painter.fillRect(rx0, max(0, ry0 - th), tw, th, QColor(*color))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(rx0, max(fm.ascent(), ry0 - fm.descent()), tag)

        # Live selection overlays.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.tool == "rect" and self._rect_start and self._rect_end:
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            x0 = int(self._rect_start.x() * self.zoom)
            y0 = int(self._rect_start.y() * self.zoom)
            x1 = int(self._rect_end.x() * self.zoom)
            y1 = int(self._rect_end.y() * self.zoom)
            painter.drawRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        elif self.tool == "lasso" and len(self._lasso_pts) >= 2:
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            poly = QPolygon([QPoint(int(x * self.zoom), int(y * self.zoom))
                             for x, y in self._lasso_pts])
            painter.drawPolyline(poly)
        elif self.tool == "brush" and self._cursor_pos is not None:
            painter.setPen(QPen(QColor(0, 200, 255), 1))
            r = int(self.brush_radius * self.zoom)
            painter.drawEllipse(self._cursor_pos, r, r)
