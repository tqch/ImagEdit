# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""AI human segmentation using YOLO (person detection) + SAM (mask generation).

Models are loaded lazily and cached, so importing this module is cheap and the
GUI starts instantly. If `ultralytics` (or the model weights) are unavailable,
`is_available()` returns False and the GUI hides/greys-out the AI features
instead of crashing.

Weights are downloaded automatically by ultralytics on first use:
  - yolov8n.pt   (~6 MB)   person detection
  - mobile_sam.pt (~40 MB) lightweight SAM for box-prompted masks
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import extensions

# If a previously-installed AI extension pack exists, load it now so the heavy
# torch/ultralytics libraries become importable before is_available() runs.
extensions.auto_activate()

_PERSON_CLASS_ID = 0  # COCO class id for "person"

# Module-level caches so models load only once per process.
_yolo_model = None
_sam_model = None
_import_error: str | None = None


@dataclass
class Subject:
    """A single detected person."""
    index: int
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels
    mask: np.ndarray                # boolean (H, W) silhouette
    score: float                    # detection confidence
    label: str = "person"


def is_available() -> bool:
    """True if ultralytics can be imported (weights download on demand)."""
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        global _import_error
        _import_error = str(exc)
        return False


def unavailable_reason() -> str:
    return _import_error or (
        "Body segmentation (YOLO+SAM) needs the AI extension pack. "
        "Click 'Enable people detection' to load it, or "
        "`pip install ultralytics torch` in a dev install."
    )


def available_devices() -> list[str]:
    """Return the compute devices the UI should offer, e.g. ['auto','mps','cpu'].

    'mps' = Apple Silicon GPU (Metal), 'cuda' = NVIDIA GPU. 'auto' lets the
    library pick the best available. Detection is best-effort and never raises.
    """
    devices = ["auto"]
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:
        pass
    devices.append("cpu")
    return devices


def resolve_device(device: str = "auto") -> str:
    """Turn 'auto' into a concrete device string, preferring GPU when present."""
    if device and device != "auto":
        return device
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _load_models(yolo_weights: str = "yolov8n.pt",
                 sam_weights: str = "mobile_sam.pt"):
    """Lazily load and cache the YOLO and SAM models."""
    global _yolo_model, _sam_model
    from ultralytics import YOLO, SAM  # imported here to keep startup light

    # Prefer weight files bundled in the extension pack (offline, no download).
    if _yolo_model is None:
        _yolo_model = YOLO(extensions.weight_path(yolo_weights))
    if _sam_model is None:
        _sam_model = SAM(extensions.weight_path(sam_weights))
    return _yolo_model, _sam_model


def detect_people(image_rgb: np.ndarray,
                  conf: float = 0.35,
                  yolo_weights: str = "yolov8n.pt",
                  sam_weights: str = "mobile_sam.pt",
                  device: str = "auto",
                  progress=None) -> list[Subject]:
    """Detect each person and return per-subject silhouette masks.

    Pipeline: YOLO finds person bounding boxes, then SAM is prompted with each
    box to produce a precise instance silhouette. Falls back to filled boxes if
    SAM produces nothing for a given person.

    `device` selects the compute backend ("auto", "mps", "cuda", "cpu").
    `progress` is an optional callable(str) for status updates.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) RGB image")

    def _say(msg: str) -> None:
        if progress:
            progress(msg)

    dev = resolve_device(device)
    _say(f"Loading models ({dev})...")
    yolo, sam = _load_models(yolo_weights, sam_weights)

    h, w = image_rgb.shape[:2]
    # ultralytics expects BGR or accepts RGB arrays; pass RGB consistently.
    _say("Detecting people...")
    det = yolo.predict(image_rgb, conf=conf, classes=[_PERSON_CLASS_ID],
                       device=dev, verbose=False)[0]

    boxes = []
    scores = []
    if det.boxes is not None and len(det.boxes) > 0:
        xyxy = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), sc in zip(xyxy, confs):
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
            scores.append(float(sc))

    if not boxes:
        _say("No people detected.")
        return []

    _say(f"Segmenting {len(boxes)} subject(s)...")
    subjects: list[Subject] = []
    # SAM accepts all boxes at once via the bboxes prompt.
    try:
        sam_res = sam.predict(image_rgb, bboxes=[list(b) for b in boxes],
                              device=dev, verbose=False)[0]
        sam_masks = None
        if sam_res.masks is not None:
            sam_masks = sam_res.masks.data.cpu().numpy()  # (N, H, W)
    except Exception:
        sam_masks = None

    for i, (box, sc) in enumerate(zip(boxes, scores)):
        mask = None
        if sam_masks is not None and i < len(sam_masks):
            m = sam_masks[i]
            if m.shape != (h, w):
                import cv2
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
            mask = m.astype(bool)
        if mask is None or not mask.any():
            # Fallback: use the bounding box as the mask.
            mask = np.zeros((h, w), dtype=bool)
            x0, y0, x1, y1 = box
            mask[max(0, y0):y1, max(0, x0):x1] = True
        subjects.append(Subject(index=i + 1, box=box, mask=mask, score=sc))

    _say(f"Found {len(subjects)} subject(s).")
    return subjects
