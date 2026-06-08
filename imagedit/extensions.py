# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tianqi Chen
# This file is part of ImagEdit. See the LICENSE file for full terms.
"""Optional AI extension pack loader.

The shipped Windows build is the "lite" variant: it has no torch / ultralytics,
so YOLO+SAM body segmentation is disabled. Interested users can download a
separate **AI extension pack** and point the app at it. The pack is a folder
(or a .zip / .tar.* / .7z archive of one) laid out as:

    <pack>/
        manifest.json        # name, version, python, platform
        libs/                # torch, torchvision, ultralytics + deps
        models/              # yolov8n.pt, mobile_sam.pt (weights, optional)

Activating a pack prepends `libs/` to sys.path (so `import torch, ultralytics`
works) and records `models/` so weights load from disk instead of downloading.
The chosen pack is remembered, so it auto-activates on the next launch.

Because torch wheels are specific to a Python version and OS/arch, the pack
must match the running interpreter; the manifest is checked and a mismatch is
reported (activation still proceeds, in case the user knows better).
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

_models_dir: Path | None = None
_active_root: Path | None = None


# ----- locations ------------------------------------------------------------
def data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ImagEdit"


def _config_path() -> Path:
    return data_dir() / "config.json"


def _extract_target() -> Path:
    return data_dir() / "ai-extension"


def _load_config() -> dict:
    p = _config_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_config(root: Path) -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    cfg["extension_root"] = str(root)
    _config_path().write_text(json.dumps(cfg, indent=2))


def forget() -> None:
    """Forget the saved pack (does not delete files)."""
    cfg = _load_config()
    cfg.pop("extension_root", None)
    _config_path().write_text(json.dumps(cfg, indent=2))


# ----- platform check -------------------------------------------------------
def current_platform_tag() -> str:
    return f"py{sys.version_info.major}{sys.version_info.minor}-" \
           f"{sysconfig.get_platform()}"


def manifest_mismatch(root: Path) -> str | None:
    """Return a human-readable warning if the pack targets a different runtime."""
    mpath = root / "manifest.json"
    if not mpath.exists():
        return None
    try:
        man = json.loads(mpath.read_text())
    except Exception:
        return None
    want_py = man.get("python")
    want_plat = man.get("platform")
    cur_py = f"{sys.version_info.major}.{sys.version_info.minor}"
    cur_plat = sysconfig.get_platform()
    problems = []
    if want_py and not cur_py.startswith(str(want_py)):
        problems.append(f"Python {want_py} (you have {cur_py})")
    if want_plat and want_plat != cur_plat:
        problems.append(f"platform {want_plat} (you have {cur_plat})")
    if problems:
        return ("This AI pack was built for " + " and ".join(problems) +
                ". It may fail to load.")
    return None


# ----- archive handling -----------------------------------------------------
def _is_archive(p: Path) -> bool:
    s = p.name.lower()
    return s.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2",
                       ".tbz2", ".tar.xz", ".txz", ".7z"))


def _extract(archive: Path, dest: Path, progress=None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if progress:
        progress(f"Extracting {archive.name}...")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    elif name.endswith(".7z"):
        try:
            import py7zr
        except Exception as exc:
            raise RuntimeError(
                "Reading .7z packs needs the 'py7zr' package, which isn't "
                "available in this build. Please use the .zip pack instead."
            ) from exc
        with py7zr.SevenZipFile(archive, "r") as z:
            z.extractall(dest)
    else:  # any tar.*
        mode = "r:*"
        with tarfile.open(archive, mode) as t:
            t.extractall(dest)


def _find_root(base: Path) -> Path:
    """A pack root contains 'libs'. Handle archives with a single top folder."""
    if (base / "libs").is_dir():
        return base
    subdirs = [d for d in base.iterdir() if d.is_dir()] if base.is_dir() else []
    for d in subdirs:
        if (d / "libs").is_dir():
            return d
    return base


# ----- activation -----------------------------------------------------------
def resolve(source: str | os.PathLike, progress=None) -> Path:
    """Turn a folder or archive path into an extracted pack root."""
    p = Path(source).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.is_dir():
        return _find_root(p)
    if _is_archive(p):
        dest = _extract_target()
        # Clear any previous extraction to avoid stale mixes.
        if dest.exists():
            import shutil
            shutil.rmtree(dest, ignore_errors=True)
        _extract(p, dest, progress=progress)
        return _find_root(dest)
    raise ValueError(f"Not a folder or supported archive: {p}")


def activate(source: str | os.PathLike, progress=None,
             remember: bool = True) -> Path:
    """Activate a pack: add libs to sys.path, register models dir, save config."""
    global _models_dir, _active_root
    root = resolve(source, progress=progress)
    libs = root / "libs"
    if not libs.is_dir():
        raise ValueError(
            f"{root} is not a valid AI pack (no 'libs' folder inside).")
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))
    importlib.invalidate_caches()
    models = root / "models"
    _models_dir = models if models.is_dir() else None
    _active_root = root
    if remember:
        _save_config(root)
    if progress:
        progress("Extension activated.")
    return root


def auto_activate() -> bool:
    """Silently activate a previously-saved or env-specified pack, if present."""
    if is_active():
        return True
    candidates = []
    env = os.environ.get("IMAGEDIT_EXT_PACK")
    if env:
        candidates.append(env)
    saved = _load_config().get("extension_root")
    if saved:
        candidates.append(saved)
    for c in candidates:
        try:
            if Path(c).expanduser().exists():
                activate(c, remember=False)
                return True
        except Exception:
            continue
    return False


def is_active() -> bool:
    """True once torch + ultralytics are importable (pack loaded or dev install)."""
    try:
        import importlib.util
        return (importlib.util.find_spec("torch") is not None and
                importlib.util.find_spec("ultralytics") is not None)
    except Exception:
        return False


def models_dir() -> Path | None:
    return _models_dir


def active_root() -> Path | None:
    return _active_root


def weight_path(name: str) -> str:
    """Local path to a weight file inside the pack, or the bare name (download)."""
    if _models_dir is not None:
        cand = _models_dir / name
        if cand.exists():
            return str(cand)
    return name
