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
    pkgs = ["torch", "torchvision", "ultralytics"]
    cmd = [sys.executable, "-m", "pip", "install", "--target", str(libs)]
    if cpu_only:
        # CPU wheels keep the pack much smaller; people segmentation still runs.
        cmd += ["--index-url", "https://download.pytorch.org/whl/cpu",
                "torch", "torchvision"]
        run(cmd)
        run([sys.executable, "-m", "pip", "install", "--target", str(libs),
             "ultralytics"])
    else:
        run(cmd + pkgs)


def fetch_weights(models: Path) -> None:
    models.mkdir(parents=True, exist_ok=True)
    # Use ultralytics to download into models/, regardless of where it installed.
    sys.path.insert(0, str(models.parent / "libs"))
    from ultralytics import YOLO, SAM  # noqa: E402
    for name, ctor in [("yolov8n.pt", YOLO), ("mobile_sam.pt", SAM)]:
        print(f"Fetching {name}...", flush=True)
        ctor(name)  # downloads to CWD
        src = Path.cwd() / name
        if src.exists():
            shutil.move(str(src), str(models / name))


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
    fetch_weights(out / "models")
    write_manifest(out, args.version)
    print(f"Pack built at {out}")
    if not args.no_zip:
        zp = make_zip(out, args.version)
        print(f"Archive: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
