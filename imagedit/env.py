# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Environment setup, imported before any torch/ultralytics code runs.

This bakes in sensible defaults so the app "just works" on Apple Silicon:

- PYTORCH_ENABLE_MPS_FALLBACK=1 lets any op not yet implemented on the Apple
  GPU (MPS) fall back to CPU instead of raising. Without it, some SAM/YOLO ops
  can crash on Metal.

`setdefault` is used throughout so anything you export in your shell still wins.
Import this module as early as possible (it is imported by `imagedit/__init__`).
"""
from __future__ import annotations

import os

# Defaults applied unless the user already set them in the environment.
_DEFAULTS = {
    # Allow MPS -> CPU fallback for unsupported ops (Apple Silicon).
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    # Keep ultralytics from phoning home / printing update nags.
    "YOLO_VERBOSE": "False",
}


def apply() -> None:
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)


# Apply on import so simply importing the package is enough.
apply()
