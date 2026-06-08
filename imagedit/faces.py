# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Face detection using OpenCV's YuNet (cv2.FaceDetectorYN).

Unlike the YOLO+SAM pipeline this needs no torch/GPU — it runs in milliseconds
on CPU and the ONNX model is ~230 KB. The model ships bundled in
imagedit/assets/models/; if missing it is downloaded once from the OpenCV zoo.

Faces are returned as `segmentation.Subject` objects (label="face") with an
elliptical mask, so the existing overlay / per-subject redaction UI works
unchanged.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .segmentation import Subject

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/" + MODEL_NAME)

# Expand the raw face box by this fraction so chins/foreheads aren't clipped.
BOX_EXPAND = 0.22
# Downscale very large images for detection speed; boxes are scaled back.
MAX_SIDE = 1280

_detector = None
_detector_size: tuple[int, int] | None = None


def _model_path() -> Path:
    """Locate the YuNet model, downloading it once if necessary."""
    bundled = Path(__file__).resolve().parent / "assets" / "models" / MODEL_NAME
    if bundled.exists():
        return bundled
    cache = Path(os.environ.get("IMAGEDIT_CACHE",
                                Path.home() / ".cache" / "imagedit"))
    cached = cache / MODEL_NAME
    if not cached.exists():
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, cached)  # one-time, ~230 KB
    return cached


def is_available() -> bool:
    return hasattr(cv2, "FaceDetectorYN")


def _get_detector(size: tuple[int, int]):
    global _detector, _detector_size
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(
            str(_model_path()), "", size, score_threshold=0.6,
            nms_threshold=0.3, top_k=500)
        _detector_size = size
    elif _detector_size != size:
        _detector.setInputSize(size)
        _detector_size = size
    return _detector


def _ellipse_mask(shape: tuple[int, int],
                  box: tuple[int, int, int, int]) -> np.ndarray:
    """Elliptical mask inscribed in the (expanded) face box."""
    h, w = shape
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax, ay = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, thickness=-1)
    return mask.astype(bool)


def detect_faces(image_rgb: np.ndarray, conf: float = 0.6,
                 progress=None) -> list[Subject]:
    """Detect faces and return Subjects with elliptical silhouette masks."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB image")
    if not is_available():
        raise RuntimeError("This OpenCV build lacks FaceDetectorYN; "
                           "update opencv-python (>=4.8).")
    if progress:
        progress("Detecting faces...")

    h, w = image_rgb.shape[:2]
    scale = 1.0
    det_img = image_rgb
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        det_img = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
    dh, dw = bgr.shape[:2]
    detector = _get_detector((dw, dh))
    detector.setScoreThreshold(conf)
    _, found = detector.detect(bgr)

    subjects: list[Subject] = []
    if found is None:
        return subjects
    for i, f in enumerate(found):
        x, y, bw, bh = (float(v) / scale for v in f[:4])
        score = float(f[-1])
        # Expand the box so the redaction fully covers the head area.
        ex, ey = bw * BOX_EXPAND, bh * BOX_EXPAND
        x0 = int(max(0, x - ex))
        y0 = int(max(0, y - ey))
        x1 = int(min(w, x + bw + ex))
        y1 = int(min(h, y + bh + ey))
        box = (x0, y0, x1, y1)
        subjects.append(Subject(index=i + 1, box=box,
                                mask=_ellipse_mask((h, w), box),
                                score=score, label="face"))
    if progress:
        progress(f"Found {len(subjects)} face(s).")
    return subjects
