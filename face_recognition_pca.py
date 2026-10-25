"""
face_recognition_pca.py - FACE RECOGNITION (deciding WHO a detected face is).

Eigenfaces pipeline
-------------------
  face vector (image_size**2 pixels)
        |  PCA.transform   (project onto the top-k eigenfaces)
        v
  k-dimensional "face-space" coordinates
        |  KNN (K nearest training faces, Euclidean distance)
        v
  predicted person + distance to the closest training face of that person
        |  distance > threshold ?
        v
  "Unknown"  (open-set rejection)

PCA is fitted ONLY on training faces; test faces are just projected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import face_detection as fd

MODEL_FORMAT_VERSION = 1


class ModelLoadError(RuntimeError):
    """The saved model is missing, corrupt or from an incompatible version."""


@dataclass
class Prediction:
    label: int            # index into names (best guess, even if rejected)
    best_guess: str       # name of the best guess
    name: str             # final answer: best_guess or "Unknown"
    distance: float       # distance to nearest training face of the best guess
    match_pct: float      # heuristic 0-100 score (NOT a probability)
    is_known: bool


class PCAFaceRecognizer:
    """PCA (eigenfaces) + KNN classifier with unknown-person rejection."""

    def __init__(self, n_components: int = 50, n_neighbors: int = 3,
                 image_size: int = 64, random_state: int = 42):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.image_size = int(image_size)
        self.random_state = int(random_state)
        self.pca: PCA | None = None
        self.knn: KNeighborsClassifier | None = None
        self.names: list[str] = []
        self.y_train: np.ndarray | None = None
        self.z_train: np.ndarray | None = None
        self.auto_threshold: float | None = None
        self.meta: dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, names: list[str]) -> "PCAFaceRecognizer":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D and have the same length as y.")
        if len(X) < 2:
            raise ValueError("Need at least 2 training faces.")
        k_comp = max(1, min(self.n_components, len(X) - 1, X.shape[1]))
        self.pca = PCA(n_components=k_comp, whiten=False, random_state=self.random_state)
        self.z_train = self.pca.fit_transform(X)            # fit on TRAIN data only
        k_nn = max(1, min(self.n_neighbors, len(X)))
        self.knn = KNeighborsClassifier(n_neighbors=k_nn, weights="distance",
                                        metric="euclidean").fit(self.z_train, y)
        self.names = list(names)
        self.y_train = y
        return self

    # ------------------------------------------------------------- inference
    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return self.pca.transform(X)

    def nearest_distance_any(self, X: np.ndarray) -> np.ndarray:
        """Distance of each face to its closest training face (any person)."""
        dist, _ = self.knn.kneighbors(self.transform(X), n_neighbors=1)
        return dist[:, 0]

    def distance_to_class(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Distance of each face to the closest training face of the GIVEN label."""
        Z = self.transform(X)
        out = np.full(len(Z), np.inf)
        for i, (z, c) in enumerate(zip(Z, labels)):
            mask = self.y_train == c
            if mask.any():
                out[i] = np.linalg.norm(self.z_train[mask] - z, axis=1).min()
        return out

    def predict(self, X: np.ndarray, threshold: float | None = None) -> list[Prediction]:
        """
        Classify face vectors. `threshold=None` or inf -> never reject.
        """
        self._check_fitted()
        Z = self.transform(X)
        thr = np.inf if threshold is None else float(threshold)
        k = self.knn.n_neighbors
        dists, idx = self.knn.kneighbors(Z, n_neighbors=k)
        labels = self.knn.predict(Z)
        out: list[Prediction] = []
        for row in range(len(Z)):
            lab = int(labels[row])
            same = self.y_train[idx[row]] == lab
            d = float(dists[row][same].min()) if same.any() else float(dists[row][0])
            known = d <= thr
            if np.isfinite(thr) and thr > 0:
                pct = float(np.clip(1.0 - d / (2.0 * thr), 0.0, 1.0) * 100.0)
            else:
                pct = 100.0
            guess = self.names[lab]
            out.append(Prediction(lab, guess, guess if known else "Unknown", d, pct, known))
        return out

    # ----------------------------------------------------------- visualising
    def mean_face(self) -> np.ndarray:
        return self.pca.mean_.reshape(self.image_size, self.image_size)

    def eigenfaces(self, n: int = 12) -> list[np.ndarray]:
        n = min(n, len(self.pca.components_))
        return [c.reshape(self.image_size, self.image_size) for c in self.pca.components_[:n]]

    @staticmethod
    def to_uint8(arr: np.ndarray) -> np.ndarray:
        """Stretch any float image to 0-255 for display."""
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-12:
            return np.zeros(arr.shape, dtype=np.uint8)
        return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

    def reconstruct(self, vector: np.ndarray, k: int) -> np.ndarray:
        """Rebuild a face from only the first k eigenfaces."""
        k = max(1, min(k, len(self.pca.components_)))
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1) - self.pca.mean_
        coeff = v @ self.pca.components_[:k].T
        recon = coeff @ self.pca.components_[:k] + self.pca.mean_
        return np.clip(recon, 0, 1).reshape(self.image_size, self.image_size)

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Save atomically (write temp file, then replace)."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT_VERSION,
            "n_components": self.n_components, "n_neighbors": self.n_neighbors,
            "image_size": self.image_size, "random_state": self.random_state,
            "pca": self.pca, "knn": self.knn, "names": self.names,
            "y_train": self.y_train, "z_train": self.z_train,
            "auto_threshold": self.auto_threshold, "meta": self.meta,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(payload, tmp, compress=3)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "PCAFaceRecognizer":
        path = Path(path)
        if not path.is_file():
            raise ModelLoadError("No trained model found. Train the model first.")
        try:
            p = joblib.load(path)
            if not isinstance(p, dict) or p.get("format") != MODEL_FORMAT_VERSION:
                raise ModelLoadError("Model file has an unknown format - please retrain.")
            obj = cls(p["n_components"], p["n_neighbors"], p["image_size"], p["random_state"])
            obj.pca, obj.knn, obj.names = p["pca"], p["knn"], list(p["names"])
            obj.y_train, obj.z_train = p["y_train"], p["z_train"]
            obj.auto_threshold, obj.meta = p.get("auto_threshold"), p.get("meta", {})
            obj.meta["saved_at"] = p.get("saved_at")
            return obj
        except ModelLoadError:
            raise
        except Exception as exc:      # corrupt file, scikit-learn version change ...
            raise ModelLoadError(
                f"Could not load the saved model ({type(exc).__name__}: {exc}). "
                "It may be corrupt or created with another scikit-learn version - "
                "retrain the model."
            ) from exc

    def _check_fitted(self) -> None:
        if self.pca is None or self.knn is None:
            raise RuntimeError("The model is not trained yet.")


# --------------------------------------------------------------------------
# Detection + recognition glue (used by the app and the desktop scripts)
# --------------------------------------------------------------------------
KNOWN_COLOR = (0, 170, 0)       # BGR green
UNKNOWN_COLOR = (0, 0, 220)     # BGR red
DETECT_COLOR = (200, 120, 0)    # BGR blue (detection only)


def analyse_image(bgr: np.ndarray, detector: fd.FaceDetector,
                  model: PCAFaceRecognizer | None = None,
                  threshold: float | None = None):
    """
    Detect every face in `bgr`; if a model is given also recognise each one.
    Returns (annotated_bgr, records) where records is a list of dicts.
    """
    gray = fd.to_gray(bgr)
    boxes = detector.detect(gray)
    annotated = bgr.copy()
    records: list[dict] = []
    if not boxes:
        return annotated, records

    preds: list[Prediction] = []
    if model is not None:
        vectors = np.stack([fd.preprocess_face(fd.crop(gray, b), model.image_size) for b in boxes])
        preds = model.predict(vectors, threshold)

    for i, box in enumerate(boxes):
        if model is None:
            fd.draw_label(annotated, box, f"Face {i + 1}", DETECT_COLOR)
            records.append({"Face": i + 1, "Box (x,y,w,h)": str(box)})
        else:
            p = preds[i]
            label = f"{p.name} ({p.distance:.1f})"
            fd.draw_label(annotated, box, label, KNOWN_COLOR if p.is_known else UNKNOWN_COLOR)
            records.append({
                "Face": i + 1, "Identity": p.name, "Closest match": p.best_guess,
                "Distance": round(p.distance, 3), "Match score %": round(p.match_pct, 1),
                "Box (x,y,w,h)": str(box),
            })
    return annotated, records
