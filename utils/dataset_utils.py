"""
utils/dataset_utils.py - everything that touches the dataset/ folder.

Layout (one folder per registered person):
    dataset/
        Alice/ 3fa9c1...jpg  b72e0d...jpg ...
        Bob/   ...
App-saved files are named after the first 16 hex chars of the SHA-1 of the
original upload, which makes duplicate detection trivial.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import numpy as np

import config
from utils.image_utils import encode_jpeg, resize_max_side, sha1_hex

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$")
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}


def validate_person_name(name: str) -> tuple[str, str | None]:
    """Return (clean_name, error). Prevents path traversal and illegal names."""
    clean = (name or "").strip()
    clean = re.sub(r"\s+", " ", clean)
    if not clean:
        return clean, "Please enter a name."
    if not NAME_RE.match(clean):
        return clean, ("Use 1-40 characters: letters, digits, spaces, '_' or '-' "
                       "(must start with a letter or digit).")
    if clean.upper() in RESERVED:
        return clean, "That name is reserved by Windows. Please choose another."
    return clean, None


def person_path(name: str) -> Path:
    """Folder of a person; guaranteed to stay inside dataset/."""
    path = (config.DATASET_DIR / name).resolve()
    if config.DATASET_DIR.resolve() not in path.parents:
        raise ValueError("Invalid person name.")
    return path


def list_people() -> list[str]:
    config.ensure_dirs()
    return sorted(p.name for p in config.DATASET_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("."))


def list_images(name: str) -> list[Path]:
    folder = person_path(name)
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS)


def dataset_summary() -> dict[str, int]:
    """{person: number_of_image_files}"""
    return {p: len(list_images(p)) for p in list_people()}


def find_duplicate(file_hash: str) -> str | None:
    """Return the person already owning this exact photo, else None."""
    for person in list_people():
        if (person_path(person) / f"{file_hash}.jpg").exists():
            return person
    return None


def save_face_image(name: str, bgr: np.ndarray, original_bytes: bytes) -> tuple[str, Path | None]:
    """
    Store a validated photo for `name`.
    Returns ("saved", path) or ("duplicate:<owner>", None).
    """
    file_hash = sha1_hex(original_bytes)[:16]
    owner = find_duplicate(file_hash)
    if owner is not None:
        return f"duplicate:{owner}", None
    folder = person_path(name)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{file_hash}.jpg"
    target.write_bytes(encode_jpeg(resize_max_side(bgr, config.MAX_STORED_SIDE)))
    return "saved", target


def delete_person(name: str) -> int:
    """Delete a person's folder. Returns the number of images removed."""
    folder = person_path(name)
    if not folder.is_dir():
        return 0
    count = len(list_images(name))
    shutil.rmtree(folder)
    return count


def dataset_signature() -> str:
    """
    Fingerprint of the current dataset (file names + sizes). Stored in the
    trained model so the app can tell when the model is out of date.
    """
    digest = hashlib.sha1()
    for person in list_people():
        for f in list_images(person):
            digest.update(f"{person}/{f.name}:{f.stat().st_size};".encode("utf-8"))
    return digest.hexdigest()
