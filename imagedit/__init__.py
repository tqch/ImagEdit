# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""ImagEdit - a lightweight GIMP-style image redaction tool.

Add mosaic or blur to user-selected regions, paint redaction with a brush,
or let AI detect people and redact each subject individually.
"""

__version__ = "0.1.0"

# Apply environment defaults (e.g. MPS CPU-fallback) before any torch import.
from . import env as _env  # noqa: E402,F401
