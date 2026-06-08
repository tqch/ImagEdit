# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Main application window for ImagEdit.

Ties together the layered canvas, the tool settings (which double as a live
editor for the selected layer), the Layers panel, file open/save, undo/redo,
and the AI "detect people/faces then redact each subject" workflow.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QButtonGroup, QRadioButton,
    QGroupBox, QFileDialog, QMessageBox, QDockWidget, QFrame, QCheckBox,
    QSplashScreen, QToolBar, QColorDialog, QMenu, QLineEdit,
)

from .canvas import Canvas
from . import segmentation
from . import faces
from . import extensions

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

ASSETS = Path(__file__).resolve().parent / "assets"

# Distinct colors for AI subject overlays (cycled if more subjects).
SUBJECT_PALETTE = [
    (255, 99, 71), (50, 205, 50), (30, 144, 255), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 140, 0), (240, 98, 146),
]

# Default values for the tool sliders (the Reset-parameters button).
SLIDER_DEFAULTS = {"block": 12, "strength": 15, "brush": 25, "feather": 0}


def load_image_rgb(path: str) -> np.ndarray:
    if _HAVE_PIL:
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str, arr: np.ndarray) -> None:
    if _HAVE_PIL:
        Image.fromarray(arr).save(path)
    else:
        import cv2
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


class DetectWorker(QThread):
    """Runs the (slow) AI body segmentation off the UI thread."""
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, image: np.ndarray, device: str = "auto"):
        super().__init__()
        self._image = image
        self._device = device

    def run(self):
        try:
            subjects = segmentation.detect_people(
                self._image, device=self._device, progress=self.progress.emit)
            self.finished_ok.emit(subjects)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class ExtensionWorker(QThread):
    """Extracts + activates an AI extension pack off the UI thread."""
    finished_ok = pyqtSignal(str)   # warning message ("" if none)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, source: str):
        super().__init__()
        self._source = source

    def run(self):
        try:
            root = extensions.activate(self._source, progress=self.progress.emit)
            warn = extensions.manifest_mismatch(root) or ""
            self.finished_ok.emit(warn)
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


class SubjectRow(QFrame):
    """A detected subject: legend swatch + label + style + Redact + Reset."""
    def __init__(self, subject, color, on_apply, on_reset, on_hover):
        super().__init__()
        self.subject = subject
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(
            f"background-color: rgb({color[0]},{color[1]},{color[2]});"
            "border-radius: 3px; border: 1px solid #555;")
        lay.addWidget(swatch)
        lbl = QLabel(f"{subject.label.title()} {subject.index} ({subject.score:.0%})")
        lbl.setMinimumWidth(96)
        self.combo = QComboBox()
        self.combo.addItems(["mosaic", "blur", "fill"])
        redact_btn = QPushButton("Redact")
        redact_btn.clicked.connect(
            lambda: on_apply(self.subject, self.combo.currentText()))
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Remove this subject's redaction")
        reset_btn.clicked.connect(lambda: on_reset(self.subject))
        lay.addWidget(lbl)
        lay.addWidget(self.combo)
        lay.addWidget(redact_btn)
        lay.addWidget(reset_btn)
        self.enterEvent = lambda e: on_hover(self.subject)
        self.leaveEvent = lambda e: on_hover(None)


class LayerRow(QFrame):
    """A row in the Layers panel: visibility + name + remove; click to select."""
    def __init__(self, layer, selected, on_select, on_toggle, on_remove):
        super().__init__()
        self.layer_id = layer.id
        self._on_select = on_select
        self.setFrameShape(QFrame.Shape.StyledPanel)
        if selected:
            self.setStyleSheet("background-color: #2b4a63; border: 1px solid #0bf;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        vis = QCheckBox()
        vis.setChecked(layer.visible)
        vis.setToolTip("Show / hide this redaction")
        vis.toggled.connect(lambda v: on_toggle(layer.id, v))
        name = QLabel(f"{layer.name}  ·  {layer.style}")
        rm = QPushButton("✕")
        rm.setFixedWidth(26)
        rm.setToolTip("Remove this redaction layer")
        rm.clicked.connect(lambda: on_remove(layer.id))
        lay.addWidget(vis)
        lay.addWidget(name, 1)
        lay.addWidget(rm)

    def mousePressEvent(self, event):
        self._on_select(self.layer_id)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImagEdit - mosaic & blur redaction")
        self.resize(1240, 820)
        icon_path = ASSETS / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.canvas = Canvas()
        self.canvas.status_message.connect(self._on_status)
        self.canvas.history_changed.connect(self._update_actions)
        self.canvas.layers_changed.connect(self._on_layers_changed)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.scroll)

        self._subjects: list = []
        self._subject_layers: dict[int, int] = {}  # subject.index -> layer id
        self._worker: DetectWorker | None = None
        self._current_path: str | None = None
        self._loading = False       # guard while loading layer params into UI
        self._fill_color = (0, 0, 0)

        self._build_menu()
        self._build_toolbar()
        self._build_tools_dock()
        self._build_layers_dock()
        self._build_ai_dock()
        self._build_watermark_dock()
        self.statusBar().showMessage("Open an image to begin.")
        self._update_actions()
        self._sync_template()
        self._on_layers_changed()

    # ----- menus / actions --------------------------------------------------
    def _build_menu(self):
        m = self.menuBar()
        filem = m.addMenu("&File")
        self.act_open = QAction("&Open...", self, shortcut=QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_image)
        self.act_save = QAction("&Save As...", self, shortcut=QKeySequence.StandardKey.SaveAs)
        self.act_save.triggered.connect(self.save_image)
        filem.addAction(self.act_open)
        filem.addAction(self.act_save)
        filem.addSeparator()
        quit_act = QAction("&Quit", self, shortcut=QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        filem.addAction(quit_act)

        editm = m.addMenu("&Edit")
        self.act_undo = QAction("&Undo", self, shortcut=QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.canvas.undo)
        self.act_redo = QAction("&Redo", self, shortcut=QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.canvas.redo)
        editm.addAction(self.act_undo)
        editm.addAction(self.act_redo)
        editm.addSeparator()
        self.act_reset = QAction("Reset to &Original", self)
        self.act_reset.setShortcut("Ctrl+Shift+R")
        self.act_reset.triggered.connect(self._reset_image)
        editm.addAction(self.act_reset)

        viewm = m.addMenu("&View")
        for label, fn, sc in [
            ("Zoom &In", lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25), "Ctrl+="),
            ("Zoom &Out", lambda: self.canvas.set_zoom(self.canvas.zoom / 1.25), "Ctrl+-"),
            ("&Fit to Window", self.zoom_fit, "Ctrl+0"),
            ("Actual &Size", lambda: self.canvas.set_zoom(1.0), "Ctrl+1"),
        ]:
            a = QAction(label, self)
            a.setShortcut(sc)
            a.triggered.connect(fn)
            viewm.addAction(a)

        helpm = m.addMenu("&Help")
        about = QAction("&About ImagEdit", self)
        about.triggered.connect(self._show_about)
        helpm.addAction(about)
        lic = QAction("&License", self)
        lic.triggered.connect(self._show_license)
        helpm.addAction(lic)

    def _show_about(self):
        from . import __version__
        QMessageBox.about(
            self, "About ImagEdit",
            f"<h3>ImagEdit {__version__}</h3>"
            "<p>Mosaic &amp; blur image redaction with AI subject detection.</p>"
            "<p>Copyright © 2026 Tianqi Chen<br>"
            "Licensed under the <b>GNU GPL v3 or later</b>.</p>"
            "<p>This program is free software: you can redistribute it and/or "
            "modify it under the terms of the GPL. It comes with "
            "<b>ABSOLUTELY NO WARRANTY</b>. See Help → License for details.</p>"
            "<p style='color:gray'>Built with PyQt6 (GPLv3), OpenCV, NumPy, "
            "Pillow. Optional AI pack uses PyTorch and Ultralytics YOLO "
            "(AGPL-3.0) + SAM.</p>")

    def _show_license(self):
        # Search dev-tree, frozen onedir (_internal/.. and exe dir) locations.
        here = Path(__file__).resolve().parent
        candidates = [
            here.parent / "LICENSE",                 # source checkout
            here.parent.parent / "LICENSE",          # frozen _internal/..
            Path(getattr(sys, "_MEIPASS", "")) / "LICENSE" if getattr(sys, "_MEIPASS", "") else None,
            Path(sys.executable).resolve().parent / "LICENSE",
        ]
        text = ("The GNU General Public License v3 governs this program.\n"
                "See the LICENSE file in the installation folder, or "
                "https://www.gnu.org/licenses/gpl-3.0.html")
        for license_path in candidates:
            if license_path and license_path.exists():
                try:
                    text = license_path.read_text(encoding="utf-8", errors="replace")
                    break
                except Exception:
                    continue
        dlg = QMessageBox(self)
        dlg.setWindowTitle("ImagEdit License (GNU GPL v3)")
        dlg.setText("ImagEdit is licensed under the GNU GPL v3 or later.")
        dlg.setDetailedText(text)
        dlg.exec()

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)
        for act in (self.act_open, self.act_save):
            tb.addAction(act)
        tb.addSeparator()
        for act in (self.act_undo, self.act_redo):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_reset)
        self.act_undo.setText("Undo")
        self.act_redo.setText("Redo")
        self.act_reset.setText("Reset")

    def _reset_image(self):
        if not self.canvas.has_image() or not self.canvas.can_reset():
            return
        ans = QMessageBox.question(
            self, "Reset image",
            "Remove ALL redaction layers and restore the original image?\n"
            "(You can still undo the reset.)")
        if ans == QMessageBox.StandardButton.Yes:
            self._subject_layers.clear()
            self.canvas.reset_to_original()

    # ----- tools dock -------------------------------------------------------
    def _build_tools_dock(self):
        dock = QDockWidget("Tools", self)
        panel = QWidget()
        v = QVBoxLayout(panel)

        tool_box = QGroupBox("Selection tool")
        tv = QVBoxLayout(tool_box)
        self.tool_group = QButtonGroup(self)
        for key, text in [("rect", "Rectangle"), ("lasso", "Freehand lasso"),
                          ("brush", "Brush (paint redaction)")]:
            rb = QRadioButton(text)
            rb.setProperty("tool", key)
            self.tool_group.addButton(rb)
            tv.addWidget(rb)
            if key == "rect":
                rb.setChecked(True)
        self.tool_group.buttonClicked.connect(self._on_tool_changed)
        v.addWidget(tool_box)

        self._edit_hint = QLabel()
        self._edit_hint.setWordWrap(True)
        self._edit_hint.setStyleSheet("color:#0bf; font-size:11px;")
        v.addWidget(self._edit_hint)

        style_box = QGroupBox("Redaction style")
        sv = QVBoxLayout(style_box)
        self.style_combo = QComboBox()
        self.style_combo.addItems(["mosaic", "blur", "fill"])
        self.style_combo.currentTextChanged.connect(
            lambda *_: self._on_control_changed(discrete=True))
        sv.addWidget(self.style_combo)

        fillrow = QHBoxLayout()
        self.fill_btn = QPushButton("Fill color")
        self.fill_btn.clicked.connect(self._pick_fill_color)
        fillrow.addWidget(self.fill_btn)
        sv.addLayout(fillrow)
        self.fill_auto_check = QCheckBox("Auto - match surroundings")
        self.fill_auto_check.toggled.connect(self._on_fill_auto_toggled)
        sv.addWidget(self.fill_auto_check)
        self._update_fill_btn()
        v.addWidget(style_box)

        self.block_slider, block_box = self._slider("Mosaic block size", 2, 60, 12)
        self.strength_slider, strength_box = self._slider("Blur strength", 1, 99, 15)
        self.brush_slider, brush_box = self._slider("Brush radius", 3, 150, 25)
        self.feather_slider, feather_box = self._slider(
            "Edge feather  (fill expands outward)", 0, 60, 0)
        for box in (block_box, strength_box, brush_box, feather_box):
            v.addWidget(box)

        reset_params_btn = QPushButton("Reset parameters to defaults")
        reset_params_btn.clicked.connect(self._reset_params)
        v.addWidget(reset_params_btn)

        hint = QLabel("Ctrl + wheel to zoom. Draw a region or paint to add a\n"
                      "redaction layer; sliders then edit the selected layer live.")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _slider(self, label, lo, hi, val):
        box = QGroupBox(label)
        lay = QHBoxLayout(box)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        readout = QLabel(str(val))
        readout.setFixedWidth(28)
        s.valueChanged.connect(lambda v: readout.setText(str(v)))
        s.valueChanged.connect(lambda *_: self._on_control_changed())
        s.sliderPressed.connect(self._on_slider_pressed)
        lay.addWidget(s)
        lay.addWidget(readout)
        return s, box

    # ----- layers dock ------------------------------------------------------
    def _build_layers_dock(self):
        dock = QDockWidget("Layers", self)
        panel = QWidget()
        outer = QVBoxLayout(panel)
        head = QLabel("Redaction layers (top = front). Click to select & edit.")
        head.setStyleSheet("color: gray; font-size: 11px;")
        head.setWordWrap(True)
        outer.addWidget(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._layers_container = QVBoxLayout(inner)
        self._layers_container.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _on_layers_changed(self):
        # Rebuild the layers list.
        while self._layers_container.count():
            item = self._layers_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        layers = self.canvas.layers()
        if not layers:
            empty = QLabel("No redactions yet.")
            empty.setStyleSheet("color: gray;")
            self._layers_container.addWidget(empty)
        else:
            for layer in reversed(layers):  # front layer at the top
                row = LayerRow(layer, layer.id == self.canvas.selected_id,
                               self.canvas.select_layer,
                               self.canvas.set_layer_visible,
                               self.canvas.remove_layer)
                self._layers_container.addWidget(row)
        self._layers_container.addStretch(1)

        # Load the selected layer's params into the editor controls.
        layer = self.canvas.selected_layer()
        if layer is not None:
            self._load_layer_into_controls(layer)
            self._edit_hint.setText(f"Editing: {layer.name}")
        else:
            self._edit_hint.setText("New redactions use these settings.")
        self._update_actions()

    # ----- live editing / template ------------------------------------------
    def _on_slider_pressed(self):
        # One undo step per drag gesture on the selected layer.
        if self.canvas.selected_layer() is not None:
            self.canvas.begin_edit()

    def _on_control_changed(self, discrete: bool = False):
        if self._loading:
            return
        self._sync_template()
        layer = self.canvas.selected_layer()
        if layer is not None:
            if discrete:
                self.canvas.begin_edit()
            self.canvas.update_layer(
                layer.id,
                style=self.style_combo.currentText(),
                strength=self.strength_slider.value(),
                block_size=self.block_slider.value(),
                feather=self.feather_slider.value(),
                fill_color=None if self.fill_auto_check.isChecked() else self._fill_color)

    def _sync_template(self):
        self.canvas.style = self.style_combo.currentText()
        self.canvas.block_size = self.block_slider.value()
        self.canvas.strength = self.strength_slider.value()
        self.canvas.brush_radius = self.brush_slider.value()
        self.canvas.feather = self.feather_slider.value()
        self.canvas.fill_color = self._fill_color
        self.canvas.fill_auto = self.fill_auto_check.isChecked()

    def _load_layer_into_controls(self, layer):
        self._loading = True
        self.style_combo.setCurrentText(layer.style)
        self.block_slider.setValue(layer.block_size)
        self.strength_slider.setValue(layer.strength)
        self.feather_slider.setValue(layer.feather)
        if layer.fill_color is None:
            self.fill_auto_check.setChecked(True)
        else:
            self.fill_auto_check.setChecked(False)
            self._fill_color = tuple(layer.fill_color)
            self._update_fill_btn()
        self._loading = False
        self._sync_template()

    def _pick_fill_color(self):
        color = QColorDialog.getColor(QColor(*self._fill_color), self, "Fill color")
        if color.isValid():
            self._fill_color = (color.red(), color.green(), color.blue())
            self._update_fill_btn()
            self._on_control_changed(discrete=True)

    def _update_fill_btn(self):
        r, g, b = self._fill_color
        text = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "black"
        self.fill_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {text};"
            "padding: 4px; border: 1px solid #777; border-radius: 3px;")
        self.fill_btn.setText(f"Fill color  ({r},{g},{b})")

    def _on_fill_auto_toggled(self, on: bool):
        self.fill_btn.setEnabled(not on)
        self._on_control_changed(discrete=True)

    def _on_tool_changed(self, btn):
        self.canvas.tool = btn.property("tool")
        self.canvas.new_brush_session()

    def _reset_params(self):
        layer = self.canvas.selected_layer()
        if layer is not None:
            self.canvas.begin_edit()
        self._loading = True   # set all sliders without per-step re-render
        self.block_slider.setValue(SLIDER_DEFAULTS["block"])
        self.strength_slider.setValue(SLIDER_DEFAULTS["strength"])
        self.brush_slider.setValue(SLIDER_DEFAULTS["brush"])
        self.feather_slider.setValue(SLIDER_DEFAULTS["feather"])
        self.fill_auto_check.setChecked(False)
        self._fill_color = (0, 0, 0)
        self._update_fill_btn()
        self._loading = False
        self._on_control_changed(discrete=False)  # apply once
        self.statusBar().showMessage("Parameters reset to defaults.")

    # ----- watermark dock ---------------------------------------------------
    def _build_watermark_dock(self):
        dock = QDockWidget("Watermark", self)
        panel = QWidget()
        v = QVBoxLayout(panel)

        self.wm_enable = QCheckBox("Add watermark / copyright")
        self.wm_enable.toggled.connect(self._wm_changed)
        v.addWidget(self.wm_enable)

        v.addWidget(QLabel("Text:"))
        self.wm_text = QLineEdit("© 2026 Your Name")
        self.wm_text.textChanged.connect(self._wm_changed)
        v.addWidget(self.wm_text)

        row = QHBoxLayout()
        row.addWidget(QLabel("Position:"))
        self.wm_pos = QComboBox()
        self.wm_pos.addItems(["bottom-right", "bottom-left", "top-right",
                              "top-left", "center", "tiled"])
        self.wm_pos.currentTextChanged.connect(self._wm_changed)
        row.addWidget(self.wm_pos)
        v.addLayout(row)

        self.wm_opacity, ob = self._plain_slider("Opacity", 0, 100, 50)
        self.wm_size, sb = self._plain_slider("Size (% of height)", 1, 20, 4)
        v.addWidget(ob)
        v.addWidget(sb)

        self._wm_color = (255, 255, 255)
        self.wm_color_btn = QPushButton("Text color")
        self.wm_color_btn.clicked.connect(self._pick_wm_color)
        v.addWidget(self.wm_color_btn)
        self._update_wm_color_btn()

        v.addStretch(1)
        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _plain_slider(self, label, lo, hi, val):
        box = QGroupBox(label)
        lay = QHBoxLayout(box)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        readout = QLabel(str(val))
        readout.setFixedWidth(28)
        s.valueChanged.connect(lambda x: readout.setText(str(x)))
        s.valueChanged.connect(lambda *_: self._wm_changed())
        lay.addWidget(s)
        lay.addWidget(readout)
        return s, box

    def _pick_wm_color(self):
        c = QColorDialog.getColor(QColor(*self._wm_color), self, "Watermark color")
        if c.isValid():
            self._wm_color = (c.red(), c.green(), c.blue())
            self._update_wm_color_btn()
            self._wm_changed()

    def _update_wm_color_btn(self):
        r, g, b = self._wm_color
        text = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "black"
        self.wm_color_btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {text};"
            "padding: 4px; border: 1px solid #777; border-radius: 3px;")

    def _wm_changed(self, *_):
        self.canvas.update_watermark(
            enabled=self.wm_enable.isChecked(),
            text=self.wm_text.text(),
            position=self.wm_pos.currentText(),
            opacity=self.wm_opacity.value(),
            size_pct=float(self.wm_size.value()),
            color=self._wm_color)

    # ----- AI dock ----------------------------------------------------------
    def _build_ai_dock(self):
        dock = QDockWidget("AI subjects", self)
        panel = QWidget()
        self._ai_layout = QVBoxLayout(panel)

        devrow = QHBoxLayout()
        devrow.addWidget(QLabel("Compute:"))
        self.device_combo = QComboBox()
        self._device_labels = {"auto": "Auto", "mps": "Apple GPU (MPS)",
                               "cuda": "NVIDIA GPU", "cpu": "CPU"}
        for d in segmentation.available_devices():
            self.device_combo.addItem(self._device_labels.get(d, d), userData=d)
        devrow.addWidget(self.device_combo)
        self._ai_layout.addLayout(devrow)

        self.detect_btn = QPushButton("Detect people")
        self.detect_btn.clicked.connect(self.detect_people)
        self._ai_layout.addWidget(self.detect_btn)

        # Enable button (lite build): load the optional AI extension pack.
        self.enable_btn = QPushButton("Enable people detection…")
        self.enable_btn.setToolTip(
            "Load the AI extension pack (torch + YOLO/SAM + weights) from disk")
        menu = QMenu(self.enable_btn)
        menu.addAction("From archive (.zip / .7z / .tar)…", self._enable_from_archive)
        menu.addAction("From folder…", self._enable_from_folder)
        self.enable_btn.setMenu(menu)
        self._ai_layout.addWidget(self.enable_btn)
        self._ext_worker: ExtensionWorker | None = None

        self.faces_btn = QPushButton("Detect faces")
        self.faces_btn.setToolTip("Fast CPU face detection (YuNet).")
        self.faces_btn.clicked.connect(self.detect_faces)
        self.faces_btn.setEnabled(faces.is_available())
        self._ai_layout.addWidget(self.faces_btn)

        self.ai_status = QLabel("")
        self.ai_status.setWordWrap(True)
        self.ai_status.setStyleSheet("color: gray; font-size: 11px;")
        self._ai_layout.addWidget(self.ai_status)

        self.overlay_check = QCheckBox("Show subject overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.setEnabled(False)
        self.overlay_check.toggled.connect(self.canvas.set_overlay_visible)
        self._ai_layout.addWidget(self.overlay_check)

        allrow = QHBoxLayout()
        allrow.addWidget(QLabel("All:"))
        self.all_style = QComboBox()
        self.all_style.addItems(["mosaic", "blur", "fill"])
        self.redact_all_btn = QPushButton("Redact all")
        self.redact_all_btn.clicked.connect(self.redact_all)
        self.redact_all_btn.setEnabled(False)
        allrow.addWidget(self.all_style)
        allrow.addWidget(self.redact_all_btn)
        self._ai_layout.addLayout(allrow)

        self._subjects_container = QVBoxLayout()
        self._ai_layout.addLayout(self._subjects_container)
        self._ai_layout.addStretch(1)

        self._refresh_ai_availability()

        dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _refresh_ai_availability(self):
        available = segmentation.is_available()
        self.detect_btn.setEnabled(available)
        self.enable_btn.setVisible(not available)
        if available:
            root = extensions.active_root()
            self.ai_status.setText(
                "People detection ready." if root is None
                else f"People detection ready (pack: {root.name}).")
        else:
            self.ai_status.setText(
                "Face detection works now. For people/body segmentation, "
                "load the AI extension pack →")

    # ----- AI extension pack ------------------------------------------------
    def _enable_from_archive(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select AI extension pack", "",
            "AI pack (*.zip *.7z *.tar *.tar.gz *.tgz *.tar.bz2 *.tar.xz);;"
            "All files (*)")
        if path:
            self._activate_extension(path)

    def _enable_from_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select extracted AI extension pack folder")
        if path:
            self._activate_extension(path)

    def _activate_extension(self, source: str):
        if self._ext_worker is not None:
            return
        self.enable_btn.setEnabled(False)
        self.ai_status.setText("Loading extension…")
        self._ext_worker = ExtensionWorker(source)
        self._ext_worker.progress.connect(self.ai_status.setText)
        self._ext_worker.finished_ok.connect(self._on_extension_ready)
        self._ext_worker.failed.connect(self._on_extension_failed)
        self._ext_worker.finished.connect(
            lambda: setattr(self, "_ext_worker", None))
        self._ext_worker.start()

    def _on_extension_ready(self, warning: str):
        self.enable_btn.setEnabled(True)
        if warning:
            QMessageBox.warning(self, "Extension loaded with warning", warning)
        if segmentation.is_available():
            self._refresh_ai_availability()
            self.statusBar().showMessage("AI extension activated.")
        else:
            self.ai_status.setText(
                "Pack loaded but torch/ultralytics still not importable — "
                "it may not match this app's Python/platform.")

    def _on_extension_failed(self, msg: str):
        self.enable_btn.setEnabled(True)
        self.ai_status.setText("Extension load failed.")
        QMessageBox.critical(self, "Could not load AI extension", msg)

    # ----- file ops ---------------------------------------------------------
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            arr = load_image_rgb(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.canvas.set_image(arr)
        self._current_path = path
        self._clear_subjects()
        self.zoom_fit()
        self.statusBar().showMessage(f"Loaded {path}  ({arr.shape[1]}x{arr.shape[0]})")

    def save_image(self):
        if not self.canvas.has_image():
            return
        suggested = self._current_path or "redacted.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image as", suggested,
            "PNG (*.png);;JPEG (*.jpg);;All files (*)")
        if not path:
            return
        try:
            save_image_rgb(path, self.canvas.get_image())
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.statusBar().showMessage(f"Saved {path}")

    def zoom_fit(self):
        if not self.canvas.has_image():
            return
        h, w = self.canvas.image_shape()
        vp = self.scroll.viewport().size()
        if w == 0 or h == 0:
            return
        z = min(vp.width() / w, vp.height() / h)
        self.canvas.set_zoom(max(0.05, min(1.0, z)))

    # ----- AI workflow ------------------------------------------------------
    def detect_people(self):
        if not self.canvas.has_image():
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        if self._worker is not None:
            return
        self.detect_btn.setEnabled(False)
        device = self.device_combo.currentData() or "auto"
        self.ai_status.setText("Starting detection...")
        self._worker = DetectWorker(self.canvas.get_original(), device=device)
        self._worker.progress.connect(self.ai_status.setText)
        self._worker.finished_ok.connect(self._on_detected)
        self._worker.failed.connect(self._on_detect_failed)
        self._worker.finished.connect(self._worker_done)
        self._worker.start()

    def _worker_done(self):
        self._worker = None
        self.detect_btn.setEnabled(segmentation.is_available())

    def _on_detect_failed(self, msg):
        self.ai_status.setText("Detection failed.")
        QMessageBox.critical(self, "AI detection failed", msg)

    def _on_detected(self, subjects):
        self._set_subjects(subjects, append=False)
        self.ai_status.setText(
            "No people detected." if not subjects
            else f"{len(subjects)} subject(s). Choose a style and redact.")

    def detect_faces(self):
        if not self.canvas.has_image():
            QMessageBox.information(self, "No image", "Open an image first.")
            return
        self.faces_btn.setEnabled(False)
        try:
            found = faces.detect_faces(self.canvas.get_original(),
                                       progress=self.ai_status.setText)
        except Exception as exc:
            QMessageBox.critical(self, "Face detection failed", str(exc))
            return
        finally:
            self.faces_btn.setEnabled(faces.is_available())
        if not found:
            self.ai_status.setText("No faces detected.")
            return
        self._set_subjects(found, append=True)
        self.ai_status.setText(
            f"Found {len(found)} face(s). Redact each individually below.")

    def _set_subjects(self, subjects, append: bool = False):
        existing = self._subjects if append else []
        merged = list(existing) + list(subjects)
        for i, s in enumerate(merged):
            s.index = i + 1
        self._clear_subjects()
        self._subjects = merged
        if not merged:
            return
        self.redact_all_btn.setEnabled(True)
        colors = [SUBJECT_PALETTE[i % len(SUBJECT_PALETTE)]
                  for i in range(len(merged))]
        self.canvas.set_subjects(merged, colors)
        self.overlay_check.setEnabled(True)
        self.overlay_check.setChecked(True)
        for s, c in zip(merged, colors):
            row = SubjectRow(s, c, self._redact_subject, self._reset_subject,
                             self._preview_subject)
            self._subjects_container.addWidget(row)

    def _clear_subjects(self):
        self._subjects = []
        self._subject_layers.clear()
        self.redact_all_btn.setEnabled(False)
        self.canvas.clear_subjects()
        self.overlay_check.setEnabled(False)
        while self._subjects_container.count():
            item = self._subjects_container.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _redact_subject(self, subject, style):
        self._sync_template()
        source = "face" if subject.label == "face" else "ai"
        # Replace any existing redaction for this subject.
        old = self._subject_layers.get(subject.index)
        if old is not None and self.canvas.get_layer(old) is not None:
            self.canvas.remove_layer(old)
        layer = self.canvas.add_layer(
            subject.mask.copy(), source=source, style=style,
            name=f"{subject.label.title()} {subject.index}")
        if layer is not None:
            self._subject_layers[subject.index] = layer.id
        self.statusBar().showMessage(
            f"Redacted {subject.label} {subject.index} ({style}).")

    def _reset_subject(self, subject):
        lid = self._subject_layers.pop(subject.index, None)
        if lid is not None:
            self.canvas.remove_layer(lid)
            self.statusBar().showMessage(
                f"Removed redaction for {subject.label} {subject.index}.")

    def redact_all(self):
        if not self._subjects:
            return
        style = self.all_style.currentText()
        for s in self._subjects:
            self._redact_subject(s, style)
        self.statusBar().showMessage(
            f"Redacted {len(self._subjects)} subjects ({style}).")

    def _preview_subject(self, subject):
        self.canvas.highlight_subject(None if subject is None else subject.index)

    # ----- misc -------------------------------------------------------------
    def _on_status(self, msg):
        self.statusBar().showMessage(msg)

    def _update_actions(self):
        self.act_undo.setEnabled(self.canvas.can_undo())
        self.act_redo.setEnabled(self.canvas.can_redo())
        self.act_reset.setEnabled(self.canvas.has_image() and self.canvas.can_reset())


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ImagEdit")

    icon_path = ASSETS / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    splash = None
    splash_path = ASSETS / "splash.png"
    if splash_path.exists():
        splash = QSplashScreen(QPixmap(str(splash_path)))
        splash.show()
        align = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        white = QColor(245, 245, 250)
        splash.showMessage("Loading interface...", align, white)
        app.processEvents()

    started = time.monotonic()
    win = MainWindow()

    if splash is not None:
        splash.showMessage("Ready.", align, white)
        app.processEvents()
        while time.monotonic() - started < 1.2:
            app.processEvents()
            time.sleep(0.03)

    win.show()
    if splash is not None:
        splash.finish(win)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
