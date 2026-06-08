#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Build the ImagEdit AI extension pack.

Produces a self-contained pack that the lite app can load to enable YOLO+SAM
people segmentation, then archives it. Layout:

    <out>/
        manifest.json
        libs/      (torch, torchvision, ultralytics + deps via pip --target)
        models/    (yolov8n.pt, mobile_sam.pt)

Usage:
    python packaging/build_extension.py --version 0.1.0 [--out build/ai-ext] \
        [--no-zip] [--cpu]

The pack is Python-version and OS/arch specific (torch ships compiled wheels),
so build it with the SAME Python and platform as the target app (the CI does
this on a Windows / Python 3.11 runner to match the exe).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

WEIGHTS = ["yolov8n.pt", "mobile_sam.pt"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pip_install_libs(libs: Path, cpu_only: bool) -> None:
    libs.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
            "--target", str(libs), "--upgrade"]
    if cpu_only:
        # torch + torchvision come from the CPU wheel index (that index also
        # hosts their own dependencies). ultralytics and its deps come from
        # PyPI, so it is installed in a second pass.
        run(base + ["--index-url", "https://download.pytorch.org/whl/cpu",
                    "torch", "torchvision"])
        # Keep the CPU index as an extra so a re-resolved torch stays CPU.
        run(base + ["--extra-index-url", "https://download.pytorch.org/whl/cpu",
                    "ultralytics"])
    else:
        run(base + ["torch", "torchvision", "ultralytics"])


def fetch_weights(libs: Path, models: Path) -> None:
    """Download the model weights into models/ using the pack's own libs.

    Runs in a *separate* interpreter with PYTHONPATH=libs (rather than importing
    the freshly --target-installed torch into this build process, which is
    unreliable on Windows). A download failure is a warning, not a build error:
    the app downloads any missing weights on first online use.
    """
    models.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(libs) + os.pathsep + env.get("PYTHONPATH", "")
    # Download each weight independently so one failure doesn't lose the other.
    code = (
        "from ultralytics import YOLO, SAM\n"
        "for ctor, name in [(YOLO,'yolov8n.pt'), (SAM,'mobile_sam.pt')]:\n"
        "    try:\n"
        "        ctor(name); print('ok', name)\n"
        "    except Exception as e:\n"
        "        print('FAILED', name, e)\n"
    )
    print("Fetching model weights in a subprocess...", flush=True)
    # Not check_call: a partial/failed download must not fail the whole build.
    subprocess.call([sys.executable, "-c", code], cwd=str(models), env=env)

    # Collect whatever was downloaded (ultralytics may place files in cwd).
    for name in WEIGHTS:
        if not (models / name).exists():
            found = next(Path(str(models)).rglob(name), None) \
                or next(Path.cwd().rglob(name), None)
            if found and found.resolve() != (models / name).resolve():
                shutil.move(str(found), str(models / name))
    have = [n for n in WEIGHTS if (models / n).exists()]
    print(f"Weights present in pack: {have or 'none (download on first use)'}",
          flush=True)


def write_manifest(out: Path, version: str) -> None:
    manifest = {
        "name": "imagedit-ai-extension",
        "version": version,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sysconfig.get_platform(),
        "provides": ["yolo", "sam"],
        "weights": WEIGHTS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


def make_zip(out: Path, version: str) -> Path:
    tag = f"py{sys.version_info.major}{sys.version_info.minor}-{sysconfig.get_platform()}"
    zip_path = out.parent / f"ImagEdit-AI-Extension-{version}-{tag}.zip"
    print(f"Zipping -> {zip_path}", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out.parent))
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.0.0")
    ap.add_argument("--out", default="build/ai-extension")
    ap.add_argument("--cpu", action="store_true",
                    help="install CPU-only torch (smaller pack)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pip_install_libs(out / "libs", cpu_only=args.cpu)
    fetch_weights(out / "libs", out / "models")
    write_manifest(out, args.version)
    print(f"Pack built at {out}")
    if not args.no_zip:
        zp = make_zip(out, args.version)
        print(f"Archive: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
