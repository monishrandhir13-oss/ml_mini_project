"""
config.py - central configuration for the PCA Face Recognition project.

* All paths are RELATIVE to the project folder (no hard-coded D:\\ or C:\\ paths).
* User-editable settings live in models/settings.json and are validated on load.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (everything is derived from the folder that contains this file)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
SNAPSHOT_DIR = BASE_DIR / "snapshots"

SETTINGS_PATH = MODELS_DIR / "settings.json"
MODEL_PATH = MODELS_DIR / "pca_face_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Haar cascade
# --------------------------------------------------------------------------
CASCADE_NAME = "haarcascade_frontalface_default.xml"
LOCAL_CASCADE_PATH = MODELS_DIR / CASCADE_NAME   # last-resort local copy
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/"
    "haarcascades/haarcascade_frontalface_default.xml"
)

# --------------------------------------------------------------------------
# Files / uploads
# --------------------------------------------------------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
UPLOAD_TYPES = ["jpg", "jpeg", "png"]
MAX_UPLOAD_MB = 8              # reject bigger uploads
MAX_STORED_SIDE = 800          # registered photos are down-scaled to this
MAX_PROCESS_SIDE = 1600        # images are down-scaled to this for detection
RECOMMENDED_IMAGES_PER_PERSON = 10

# --------------------------------------------------------------------------
# Settings (editable from the Settings page)
# --------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "image_size": 64,          # faces are resized to image_size x image_size
    "n_components": 50,        # number of eigenfaces (PCA components)
    "n_neighbors": 3,          # K of the KNN classifier
    "test_size": 0.25,         # fraction held out for evaluation
    "auto_threshold": True,    # True: use threshold calibrated during training
    "threshold_scale": 1.0,    # multiplies the calibrated threshold
    "manual_threshold": 8.0,   # used when auto_threshold is False
    "scale_factor": 1.1,       # Haar cascade parameters
    "min_neighbors": 5,
    "min_face_size": 60,
    "random_state": 42,
}

# (min, max) for numeric settings - used for validation
SETTING_LIMITS = {
    "image_size": (24, 160),
    "n_components": (2, 300),
    "n_neighbors": (1, 15),
    "test_size": (0.1, 0.5),
    "threshold_scale": (0.3, 3.0),
    "manual_threshold": (0.05, 100.0),
    "scale_factor": (1.02, 1.6),
    "min_neighbors": (1, 15),
    "min_face_size": (20, 400),
    "random_state": (0, 10_000),
}


def ensure_dirs() -> None:
    """Create the dataset / models folders if they do not exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def validate_settings(raw: dict) -> dict:
    """Return a clean settings dict: unknown keys dropped, bad values reset."""
    clean = dict(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        if key not in raw:
            continue
        value = raw[key]
        try:
            if isinstance(default, bool):
                clean[key] = bool(value)
            elif isinstance(default, int):
                value = int(value)
                lo, hi = SETTING_LIMITS[key]
                clean[key] = max(int(lo), min(int(hi), value))
            elif isinstance(default, float):
                value = float(value)
                lo, hi = SETTING_LIMITS[key]
                clean[key] = max(float(lo), min(float(hi), value))
        except (TypeError, ValueError):
            clean[key] = default
    return clean


def load_settings() -> dict:
    """Load settings.json (or defaults if missing/corrupt)."""
    ensure_dirs()
    if SETTINGS_PATH.exists():
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return validate_settings(raw)
        except (OSError, json.JSONDecodeError):
            pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> dict:
    """Validate and write settings.json. Returns the cleaned settings."""
    ensure_dirs()
    clean = validate_settings(settings)
    SETTINGS_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    return clean


def effective_threshold(settings: dict, calibrated: float | None) -> float:
    """
    Distance above which a face is called "Unknown".

    auto mode   -> calibrated threshold (learned in training) * threshold_scale
    manual mode -> the value typed in the Settings page
    """
    if settings.get("auto_threshold", True) and calibrated:
        return float(calibrated) * float(settings.get("threshold_scale", 1.0))
    return float(settings.get("manual_threshold", DEFAULT_SETTINGS["manual_threshold"]))


# --------------------------------------------------------------------------
# Privacy: admin PIN protecting dataset management
# --------------------------------------------------------------------------
DEFAULT_ADMIN_PIN = "admin123"


def admin_pin() -> str:
    """PIN for the Dataset Management page. Override with FACE_APP_ADMIN_PIN."""
    return os.environ.get("FACE_APP_ADMIN_PIN", DEFAULT_ADMIN_PIN)


def using_default_pin() -> bool:
    return admin_pin() == DEFAULT_ADMIN_PIN
