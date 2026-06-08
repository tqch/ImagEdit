# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the ImagEdit Windows build (lite variant).

Build from the project root:
    pyinstaller packaging/imagedit.spec --noconfirm

Produces dist/ImagEdit/ (one-dir build: faster startup than one-file).
The lite variant excludes torch/ultralytics, so the packaged app has all
manual tools + YuNet face detection; YOLO+SAM "Detect people" is disabled
gracefully (the app already handles their absence).
"""
from pathlib import Path

ROOT = Path(SPECPATH).parent  # packaging/ -> project root

a = Analysis(
    [str(ROOT / "run.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        # App icon, splash, and the bundled YuNet face-detection model.
        (str(ROOT / "imagedit" / "assets"), "imagedit/assets"),
        # License texts (also shown via Help -> License).
        (str(ROOT / "LICENSE"), "."),
        (str(ROOT / "THIRD-PARTY-NOTICES.md"), "."),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Lite build: no AI body-segmentation stack (~2.4 GB saved).
        "torch", "torchvision", "ultralytics",
        # Unused stdlib GUIs / misc.
        "tkinter", "unittest", "pydoc",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="ImagEdit",
    debug=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window
    icon=str(ROOT / "imagedit" / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ImagEdit",
)
