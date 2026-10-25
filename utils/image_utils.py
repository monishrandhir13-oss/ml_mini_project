"""
utils/image_utils.py - safe image loading / conversion helpers.

Why not just cv2.imread(path)?
  * cv2.imread silently returns None for paths containing non-ASCII
    characters on Windows (e.g. a user folder like C:\\Users\\Abhishék\\).
  * It ignores EXIF rotation, so phone photos can arrive sideways.
We therefore read the raw bytes and decode them with Pillow.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 40_000_000   # refuse absurdly large images (decompression bombs)


class ImageLoadError(ValueError):
    """Raised when a file is not a valid / readable image."""


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes -> BGR uint8 array (OpenCV layout)."""
    if not data:
        raise ImageLoadError("The file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()                       # detects truncated / fake files
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise ImageLoadError("The image is too large (over 40 megapixels).")
            img = ImageOps.exif_transpose(img).convert("RGB")
            rgb = np.array(img)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError) as exc:
        raise ImageLoadError("Not a valid or readable image file.") from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image_file(path) -> np.ndarray:
    """Read an image from disk (works with any Unicode path) -> BGR array."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"Cannot read file: {exc}") from exc
    return load_image_from_bytes(data)


def resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Down-scale (never up-scale) so the longest side is <= max_side."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / float(longest)
    return cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def encode_jpeg(bgr: np.ndarray, quality: int = 95) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ImageLoadError("Could not encode the image as JPEG.")
    return buf.tobytes()


def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    """Streamlit / PIL expect RGB; OpenCV uses BGR."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
