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
import os
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

WEIGHTS = ["yolov8n.pt", "mobile_sam.pt"]


# Packages not needed at runtime inside the pack (slims it down a bit).
_SKIP_DIRS = {"pip", "setuptools", "wheel", "pkg_resources", "_distutils_hack",
              "__pycache__"}


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd)


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_site_packages(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Lib" / "site-packages"
    libdir = venv / "lib"
    pyX = next(libdir.glob("python*"))
    return pyX / "site-packages"


def build_libs(out: Path, cpu_only: bool) -> Path:
    """Create a venv, install the AI stack into it, and copy its site-packages
    into <out>/libs. A real venv avoids all `pip --target` quirks and ensures
    every dependency is present in the pack.
    """
    venv = out / "_venv"
    print(f"Creating build venv at {venv} ...", flush=True)
    run([sys.executable, "-m", "venv", str(venv)])
    vpy = _venv_python(venv)
    run([str(vpy), "-m", "pip", "install", "--upgrade", "pip", "wheel"])

    if cpu_only:
        # Official PyTorch CPU index (also hosts torch's own dependencies).
        run([str(vpy), "-m", "pip", "install", "--no-cache-dir",
             "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch", "torchvision"])
        # ultralytics + its deps from PyPI; keep CPU index as a fallback.
        run([str(vpy), "-m", "pip", "install", "--no-cache-dir",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu",
             "ultralytics"])
    else:
        run([str(vpy), "-m", "pip", "install", "--no-cache-dir",
             "torch", "torchvision", "ultralytics"])

    site = _venv_site_packages(venv)
    libs = out / "libs"
    print(f"Copying {site} -> {libs} ...", flush=True)
    libs.mkdir(parents=True, exist_ok=True)
    for item in site.iterdir():
        if item.name in _SKIP_DIRS or item.name.endswith(".dist-info") and \
                item.name.split("-")[0] in _SKIP_DIRS:
            continue
        dest = libs / item.name
        if item.is_dir():
            shutil.copytree(item, dest,
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(item, dest)
    return vpy  # used to download weights from the same environment


def fetch_weights(vpy: Path, models: Path) -> None:
    """Download model weights into models/ using the build venv's interpreter.

    A download failure is a warning, not a build error: the app downloads any
    missing weights on first online use.
    """
    models.mkdir(parents=True, exist_ok=True)
    code = (
        "from ultralytics import YOLO, SAM\n"
        "for ctor, name in [(YOLO,'yolov8n.pt'), (SAM,'mobile_sam.pt')]:\n"
        "    try:\n"
        "        ctor(name); print('ok', name)\n"
        "    except Exception as e:\n"
        "        print('FAILED', name, e)\n"
    )
    print("Fetching model weights ...", flush=True)
    # cwd=models so ultralytics downloads the .pt files directly into it.
    subprocess.call([str(vpy), "-c", code], cwd=str(models))
    for name in WEIGHTS:
        if not (models / name).exists():
            found = next(Path.cwd().rglob(name), None)
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
            # Never include the throwaway build venv in the archive.
            if f.is_file() and "_venv" not in f.relative_to(out.parent).parts:
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

    print(f"Building AI extension pack: python "
          f"{sys.version_info.major}.{sys.version_info.minor} on "
          f"{sysconfig.get_platform()} (cpu_only={args.cpu})", flush=True)

    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    vpy = build_libs(out, cpu_only=args.cpu)
    fetch_weights(vpy, out / "models")
    # Remove the throwaway build venv before packaging.
    shutil.rmtree(out / "_venv", ignore_errors=True)
    write_manifest(out, args.version)
    print(f"Pack built at {out}")
    if not args.no_zip:
        zp = make_zip(out, args.version)
        print(f"Archive: {zp}  ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
