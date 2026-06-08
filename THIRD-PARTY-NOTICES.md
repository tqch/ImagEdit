# Third-party licenses

ImagEdit itself is licensed under the GNU General Public License v3.0 or later
(see `LICENSE`). It builds on the following third-party components, each under
its own license:

| Component | Used for | License |
|-----------|----------|---------|
| **PyQt6** | Desktop GUI | GPL v3 or Riverbank Commercial |
| **OpenCV** (opencv-python) | Image ops, YuNet face detection | Apache-2.0 |
| **NumPy** | Array maths | BSD-3-Clause |
| **Pillow** | Image I/O, watermark/text, icon | MIT-CMU (HPND) |
| **YuNet face model** (`face_detection_yunet_2023mar.onnx`) | Face detection | MIT (OpenCV Zoo) |

## Optional AI extension pack only

These are **not** bundled in the app; they ship in the separately-downloaded AI
extension pack and are loaded at the user's choice:

| Component | Used for | License |
|-----------|----------|---------|
| **PyTorch** (torch, torchvision) | Model runtime | BSD-3-Clause |
| **Ultralytics YOLO** | Person detection | **AGPL-3.0** (or Ultralytics Enterprise) |
| **SAM / MobileSAM** weights | Person silhouette masks | Apache-2.0 |

### Note on copyleft

PyQt6's GPL option is what makes ImagEdit's GPL v3 license the natural choice:
distributing the app linked against GPL PyQt6 requires the whole program to be
GPL-compatible. The optional AI pack adds **Ultralytics YOLO (AGPL-3.0)**, whose
network-copyleft terms apply only when that pack is installed; users who deploy
ImagEdit with the AI pack as a network service should review the AGPL, and
commercial users may need an Ultralytics Enterprise license. None of this
affects the base app, which contains no AGPL code.
