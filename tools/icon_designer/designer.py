#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Icon pattern designer - choose exactly which blocks to keep on the icon's
pixelated right half (e.g. to spell out initials like "TC").

Run from the project root:
    python tools/icon_designer/designer.py

Left: editable block grid (the icon's right half). Click a cell to toggle it
between kept (white) and erased (background). The faint shapes show where the
person silhouette lies, for reference - you may paint anywhere on the grid.
Right: live preview of the resulting icon.

"Save + regenerate" writes imagedit/assets/icon_pattern.json and rebuilds
icon.png / icon.ico / splash.png; make_assets.py uses the saved pattern from
then on (delete the json to go back to random pixelation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/
import make_assets  # noqa: E402

from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

CELL = 24  # on-screen size of one editable cell
BG = QColor(36, 31, 49)
WHITE = QColor(232, 230, 240)
HINT = QColor(232, 230, 240, 60)   # faint silhouette underlay
GRIDLINE = QColor(110, 105, 130)


def pil_to_pixmap(img) -> QPixmap:
    img = img.convert("RGBA")
    qimg = QImage(img.tobytes(), img.width, img.height,
                  4 * img.width, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class GridEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cols = make_assets.GRID_COLS
        self.rows = make_assets.GRID_ROWS
        self.coverage = make_assets.silhouette_coverage()
        self.kept: set[tuple[int, int]] = set()
        self.setFixedSize(self.cols * CELL + 1, self.rows * CELL + 1)

    def set_pattern(self, cells):
        self.kept = set(map(tuple, cells))
        self.update()
        self.changed.emit()

    def pattern(self):
        return sorted(self.kept)

    def mousePressEvent(self, event):
        j = int(event.position().x()) // CELL
        i = int(event.position().y()) // CELL
        if 0 <= j < self.cols and 0 <= i < self.rows:
            cell = (j, i)
            if cell in self.kept:
                self.kept.discard(cell)
            else:
                self.kept.add(cell)
            self.update()
            self.changed.emit()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        for i in range(self.rows):
            for j in range(self.cols):
                x, y = j * CELL, i * CELL
                if (j, i) in self.kept:
                    p.fillRect(x, y, CELL, CELL, WHITE)
                elif self.coverage.get((j, i), 0) >= 0.5:
                    p.fillRect(x, y, CELL, CELL, HINT)  # silhouette hint
        p.setPen(QPen(GRIDLINE, 1))
        for i in range(self.rows + 1):
            p.drawLine(0, i * CELL, self.cols * CELL, i * CELL)
        for j in range(self.cols + 1):
            p.drawLine(j * CELL, 0, j * CELL, self.rows * CELL)


class Designer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImagEdit icon designer")

        self.grid = GridEditor()
        self.grid.changed.connect(self.refresh_preview)

        self.preview = QLabel()
        self.preview.setFixedSize(280, 280)
        self.preview.setScaledContents(True)

        load_btn = QPushButton("Load saved pattern")
        load_btn.clicked.connect(self.load_saved)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.grid.set_pattern([]))
        fill_btn = QPushButton("Fill glyph")
        fill_btn.clicked.connect(self.fill_silhouette)
        save_btn = QPushButton("Save + regenerate assets")
        save_btn.clicked.connect(self.save)

        left = QVBoxLayout()
        hint = QLabel("Click cells to toggle keep/erase.\n"
                      "Faint cells = the 像 glyph (reference only).")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addWidget(self.grid)

        right = QVBoxLayout()
        right.addWidget(QLabel("Live preview:"))
        right.addWidget(self.preview)
        for b in (load_btn, fill_btn, clear_btn, save_btn):
            right.addWidget(b)
        right.addStretch(1)

        root = QWidget()
        lay = QHBoxLayout(root)
        lay.addLayout(left)
        lay.addLayout(right)
        self.setCentralWidget(root)

        self.load_saved(silent=True)
        self.refresh_preview()

    def load_saved(self, silent=False):
        pattern = make_assets.load_pattern()
        if pattern is not None:
            self.grid.set_pattern(pattern)
        elif not silent:
            QMessageBox.information(self, "No pattern",
                                    "No saved pattern yet - design one!")

    def fill_silhouette(self):
        cells = [c for c, cov in self.grid.coverage.items() if cov >= 0.5]
        self.grid.set_pattern(cells)

    def refresh_preview(self):
        img = make_assets.make_icon(512, pattern=self.grid.pattern())
        self.preview.setPixmap(pil_to_pixmap(img))

    def save(self):
        make_assets.PATTERN_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_assets.PATTERN_PATH.write_text(
            json.dumps({"keep": self.grid.pattern()}, indent=2))
        make_assets.main()
        QMessageBox.information(
            self, "Saved",
            f"Pattern saved to {make_assets.PATTERN_PATH.name} and assets "
            "regenerated (icon.png, icon.ico, splash.png).")


def main():
    app = QApplication(sys.argv)
    win = Designer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
