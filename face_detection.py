"""
face_detection.py - FACE DETECTION (finding WHERE faces are).

Detection answers "is there a face, and where?".  It does NOT know whose face
it is - that is the job of face_recognition_pca.py.

Main fixes compared with the original scripts
---------------------------------------------
* The Haar cascade XML is located robustly (several fallbacks, optional
  download) instead of blindly using  cv2.data.haarcascades + "...xml".
* The classifier is checked with .empty(): OpenCV does NOT raise an error when
  the XML cannot be loaded, it just detects nothing / crashes later.
* Windows + non-ASCII user name: OpenCV cannot open XML files in such folders,
  so the XML is copied to an ASCII-only temp folder and loaded from there.
* One preprocessing function is used by training AND prediction.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

import config


class CascadeNotFoundError(RuntimeError):
    """Raised when the Haar cascade cannot be found or loaded."""


# --------------------------------------------------------------------------
# Locating the Haar cascade XML
# --------------------------------------------------------------------------
def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    data = getattr(cv2, "data", None)                       # 1. official way
    if data is not None and hasattr(data, "haarcascades"):
        paths.append(Path(data.haarcascades) / config.CASCADE_NAME)
    paths.append(Path(cv2.__file__).resolve().parent / "data" / config.CASCADE_NAME)  # 2.
    paths.append(config.LOCAL_CASCADE_PATH)                 # 3. copy inside project
    return paths


def _try_download() -> Path | None:
    """Last resort: download the XML (needs internet) into models/."""
    try:
        config.ensure_dirs()
        with urllib.request.urlopen(config.CASCADE_URL, timeout=15) as resp:
            data = resp.read()
        if len(data) < 100_000 or b"<opencv_storage" not in data[:2000]:
            return None
        config.LOCAL_CASCADE_PATH.write_bytes(data)
        return config.LOCAL_CASCADE_PATH
    except Exception:                                       # network problems etc.
        return None


@lru_cache(maxsize=1)
def find_cascade_path() -> Path:
    """Return the path of haarcascade_frontalface_default.xml or raise."""
    if not hasattr(cv2, "CascadeClassifier"):
        raise CascadeNotFoundError(
            f"This OpenCV build (version {cv2.__version__}) has no CascadeClassifier, "
            "so Haar face detection is unavailable. Install a 4.x build:\n"
            '    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n'
            '    pip install "opencv-python>=4.8,<5"'
        )
    for path in _candidate_paths():
        if path.is_file():
            return path
    downloaded = _try_download()
    if downloaded is not None:
        return downloaded
    raise CascadeNotFoundError(
        f"Could not find {config.CASCADE_NAME}. Fix: run\n"
        '    pip install --force-reinstall "opencv-python>=4.8,<5"\n'
        f"or download the file manually from\n    {config.CASCADE_URL}\n"
        f"and save it as\n    {config.LOCAL_CASCADE_PATH}"
    )


def _ascii_copy(src: Path) -> Path | None:
    """Copy the XML into an ASCII-only folder (Windows non-ASCII path bug)."""
    folders = [tempfile.gettempdir(), r"C:\Users\Public", r"C:\Windows\Temp"]
    for folder in folders:
        try:
            if not os.path.isdir(folder) or not str(folder).isascii():
                continue
            dst_dir = Path(tempfile.mkdtemp(prefix="haar_", dir=folder))
            dst = dst_dir / src.name
            shutil.copy(src, dst)
            return dst
        except OSError:
            continue
    return None


def load_cascade() -> "cv2.CascadeClassifier":
    """Load the classifier, with fallbacks, and verify it really loaded."""
    path = find_cascade_path()
    clf = cv2.CascadeClassifier(str(path))
    if clf.empty():
        copy = _ascii_copy(path)
        if copy is not None:
            clf = cv2.CascadeClassifier(str(copy))
    if clf.empty():
        raise CascadeNotFoundError(
            f"Found {path} but OpenCV could not load it (corrupt file or "
            "non-ASCII folder name). Re-install opencv-python or place a fresh copy at "
            f"{config.LOCAL_CASCADE_PATH}."
        )
    return clf


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
class FaceDetector:
    """Haar-cascade face detector with parameters taken from the settings."""

    def __init__(self, settings: dict | None = None):
        s = settings or config.DEFAULT_SETTINGS
        self.scale_factor = float(s.get("scale_factor", 1.1))
        self.min_neighbors = int(s.get("min_neighbors", 5))
        self.min_size = int(s.get("min_face_size", 60))
        self._clf = load_cascade()

    def detect(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return face boxes (x, y, w, h), largest first."""
        boxes = self._clf.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        if len(boxes) == 0:
            return []
        boxes = [tuple(int(v) for v in b) for b in boxes]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


# --------------------------------------------------------------------------
# Shared preprocessing (used for training AND prediction)
# --------------------------------------------------------------------------
def to_gray(bgr: np.ndarray) -> np.ndarray:
    """BGR (or already gray) -> uint8 grayscale."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def preprocess_face(gray_crop: np.ndarray, image_size: int) -> np.ndarray:
    """
    Grayscale face crop -> flat float32 vector of length image_size**2.

    resize -> histogram equalisation (reduces lighting differences)
           -> scale pixels to [0, 1] -> flatten
    """
    face = cv2.resize(gray_crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    return (face.astype(np.float32) / 255.0).ravel()


def crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return gray[max(y, 0):y + h, max(x, 0):x + w]


def extract_largest_face(bgr: np.ndarray, detector: FaceDetector, image_size: int):
    """Detect -> take the largest face -> preprocess. Returns (vector, box) or (None, None)."""
    gray = to_gray(bgr)
    boxes = detector.detect(gray)
    if not boxes:
        return None, None
    return preprocess_face(crop(gray, boxes[0]), image_size), boxes[0]


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_label(img: np.ndarray, box, text: str, color=(0, 200, 0)) -> None:
    """Draw a bounding box with a filled label above it (in place)."""
    x, y, w, h = box
    thickness = max(2, img.shape[1] // 400)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if not text:
        return
    scale = max(0.5, img.shape[1] / 1100.0)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y - th - base - 6, 0)
    cv2.rectangle(img, (x, top), (x + tw + 8, top + th + base + 6), color, -1)
    cv2.putText(img, text, (x + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)
