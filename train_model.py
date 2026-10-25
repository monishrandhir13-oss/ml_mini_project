"""
train_model.py - build the training set, evaluate honestly, save the model.

Usage (PowerShell, from the project folder):
    python train_model.py
    python train_model.py --components 40 --neighbors 1 --image-size 64

Steps
-----
1. Load every image in dataset/<Person>/, detect the largest face, crop,
   grayscale, resize, equalise, flatten  (face_detection.preprocess_face).
2. Stratified train/test split.
3. Fit PCA + KNN on the TRAIN part only (no data leakage) and evaluate on TEST.
4. Calibrate the "Unknown" distance threshold from held-out genuine matches.
5. Estimate unknown-person rejection with leave-one-person-out.
6. Re-fit on ALL faces for the final model and save it (+ metrics.json).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import sklearn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances

import config
from face_detection import FaceDetector, extract_largest_face
from face_recognition_pca import PCAFaceRecognizer
from utils import dataset_utils as ds
from utils.image_utils import ImageLoadError, load_image_file, resize_max_side


class TrainingError(RuntimeError):
    """Raised with a user-friendly message when training is impossible."""


def _noop(frac: float, msg: str) -> None:
    pass


# --------------------------------------------------------------------------
# 1. Dataset loading
# --------------------------------------------------------------------------
def load_dataset(settings: dict, progress=_noop):
    """
    Returns (X, y, names, report).
    report = {"per_person": {name: usable}, "skipped": [(file, reason), ...]}
    """
    people = ds.list_people()
    if not people:
        raise TrainingError(
            "The dataset is empty. Add people first (Dataset Management page, or put "
            "photos in dataset/<PersonName>/).")
    detector = FaceDetector(settings)          # raises CascadeNotFoundError if unusable
    size = settings["image_size"]

    files = [(p, f) for p in people for f in ds.list_images(p)]
    X, labels, names = [], [], []
    per_person: dict[str, int] = {}
    skipped: list[tuple[str, str]] = []
    index_of: dict[str, int] = {}

    for n, (person, path) in enumerate(files, start=1):
        progress(0.05 + 0.45 * n / max(len(files), 1), f"Reading faces {n}/{len(files)}")
        try:
            bgr = resize_max_side(load_image_file(path), config.MAX_PROCESS_SIDE)
        except ImageLoadError as exc:
            skipped.append((f"{person}/{path.name}", str(exc)))
            continue
        vec, _ = extract_largest_face(bgr, detector, size)
        if vec is None:
            skipped.append((f"{person}/{path.name}", "no face detected"))
            continue
        if person not in index_of:
            index_of[person] = len(names)
            names.append(person)
        X.append(vec)
        labels.append(index_of[person])
        per_person[person] = per_person.get(person, 0) + 1

    if not X:
        raise TrainingError("No usable faces were found. Check that the photos are clear, "
                            "front-facing and well lit.")
    return np.array(X, dtype=np.float32), np.array(labels), names, {
        "per_person": per_person, "skipped": skipped}


# --------------------------------------------------------------------------
# 2-6. Training + evaluation on arrays
# --------------------------------------------------------------------------
def _new_model(settings: dict) -> PCAFaceRecognizer:
    return PCAFaceRecognizer(settings["n_components"], settings["n_neighbors"],
                             settings["image_size"], settings["random_state"])


def _drop_small_classes(X, y, names, min_count=2):
    """Remove people with fewer than min_count usable faces and re-number labels."""
    counts = np.bincount(y, minlength=len(names))
    keep = [i for i, c in enumerate(counts) if c >= min_count]
    dropped = [names[i] for i, c in enumerate(counts) if c < min_count]
    mask = np.isin(y, keep)
    remap = {old: new for new, old in enumerate(keep)}
    return X[mask], np.array([remap[v] for v in y[mask]]), [names[i] for i in keep], dropped


def _leave_one_person_out(X, y, names, settings, thr, max_people=15):
    """Share of unseen-person faces that are (correctly) rejected as Unknown."""
    people = np.unique(y)
    if len(people) < 3:
        return None
    rng = np.random.RandomState(settings["random_state"])
    chosen = people if len(people) <= max_people else rng.choice(people, max_people, replace=False)
    rejected = total = 0
    for p in chosen:
        mask = y != p
        if len(np.unique(y[mask])) < 2:
            continue
        rec = _new_model(settings).fit(X[mask], y[mask], names)
        d = rec.nearest_distance_any(X[~mask])
        rejected += int((d > thr).sum())
        total += len(d)
    return rejected / total if total else None


def train_from_arrays(X, y, names, settings: dict, progress=_noop, report: dict | None = None):
    """Evaluate + build the final model. Returns (model, metrics)."""
    started = time.time()
    report = report or {"per_person": {}, "skipped": []}
    X, y, names, dropped = _drop_small_classes(X, y, names)
    warnings = [f"'{n}' ignored: needs at least 2 usable face images." for n in dropped]
    if len(names) < 2:
        raise TrainingError("Need at least 2 people with 2+ usable face images each "
                            "to train a recognition model.")

    n_classes, n_total = len(names), len(X)
    counts = np.bincount(y)
    n_test = int(np.ceil(n_total * settings["test_size"]))
    can_split = counts.min() >= 2 and n_test >= n_classes and (n_total - n_test) >= n_classes

    metrics: dict = {"evaluation": None}
    auto_thr = None
    progress(0.55, "Fitting PCA on training faces and evaluating...")

    if can_split:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=n_test, stratify=y, random_state=settings["random_state"])
        eval_model = _new_model(settings).fit(X_tr, y_tr, names)   # PCA sees TRAIN only
        preds = eval_model.predict(X_te, threshold=None)
        y_pred = np.array([p.label for p in preds])
        genuine = eval_model.distance_to_class(X_te, y_te)
        genuine = genuine[np.isfinite(genuine)]
        auto_thr = float(np.percentile(genuine, 95)) if len(genuine) else None
        thr_used = config.effective_threshold(settings, auto_thr)
        d_match = np.array([p.distance for p in preds])
        accepted = d_match <= thr_used
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0)
        progress(0.75, "Estimating unknown-person rejection...")
        lopo = _leave_one_person_out(X, y, names, settings, thr_used)
        labels_all = list(range(n_classes))
        metrics["evaluation"] = {
            "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "accuracy_with_threshold": float(np.mean((y_pred == y_te) & accepted)),
            "known_rejected_rate": float(1.0 - accepted.mean()),
            "unknown_rejection_rate": None if lopo is None else float(lopo),
            "precision_macro": float(prec), "recall_macro": float(rec), "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(y_te, y_pred, labels=labels_all).tolist(),
            "per_person": classification_report(
                y_te, y_pred, labels=labels_all, target_names=names,
                zero_division=0, output_dict=True),
            "threshold_used": float(thr_used),
        }
    else:
        warnings.append(
            "Dataset too small for a train/test split (need >= 2 images per person and "
            "enough images for the test share) - accuracy could not be measured. "
            "Add more photos per person (10+ recommended).")

    progress(0.85, "Training final model on all faces...")
    model = _new_model(settings).fit(X, y, names)
    if auto_thr is None:      # fallback: leave-one-out distance to own class
        Z = model.z_train
        D = pairwise_distances(Z)
        np.fill_diagonal(D, np.inf)
        loo = np.array([D[i][y == y[i]].min() for i in range(len(Z))])
        loo = loo[np.isfinite(loo)]
        auto_thr = float(np.percentile(loo, 95)) if len(loo) else 1.0
        warnings.append("Unknown threshold estimated from training data (no held-out set).")
    model.auto_threshold = float(max(auto_thr, 1e-6))

    cum = np.cumsum(model.pca.explained_variance_ratio_)
    metrics.update({
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - started, 2),
        "n_people": n_classes, "n_images": int(n_total),
        "names": names,
        "images_per_person": {names[i]: int(c) for i, c in enumerate(counts)},
        "skipped": report["skipped"],
        "n_components_used": int(model.pca.n_components_),
        "explained_variance": float(cum[-1]),
        "cumulative_variance": [float(v) for v in cum],
        "auto_threshold": model.auto_threshold,
        "settings": {k: settings[k] for k in
                     ("image_size", "n_components", "n_neighbors", "test_size", "random_state")},
        "warnings": warnings,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
    })
    model.meta = {"names_count": n_classes, "n_images": int(n_total),
                  "trained_at": metrics["trained_at"]}
    return model, metrics


def train_and_save(settings: dict | None = None, progress=_noop) -> dict:
    """Full pipeline: dataset folder -> trained model + metrics on disk."""
    settings = settings or config.load_settings()
    config.ensure_dirs()
    signature = ds.dataset_signature()
    progress(0.02, "Scanning dataset...")
    X, y, names, report = load_dataset(settings, progress)
    model, metrics = train_from_arrays(X, y, names, settings, progress, report)
    model.meta["dataset_signature"] = signature
    metrics["dataset_signature"] = signature
    progress(0.95, "Saving model...")
    model.save(config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    progress(1.0, "Done")
    return metrics


def load_metrics() -> dict | None:
    try:
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete_model_files() -> None:
    for p in (config.MODEL_PATH, config.METRICS_PATH):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PCA face recognition model.")
    parser.add_argument("--components", type=int)
    parser.add_argument("--neighbors", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--test-size", type=float)
    args = parser.parse_args()

    settings = config.load_settings()
    for arg, key in [(args.components, "n_components"), (args.neighbors, "n_neighbors"),
                     (args.image_size, "image_size"), (args.test_size, "test_size")]:
        if arg is not None:
            settings[key] = arg
    settings = config.validate_settings(settings)

    def cli_progress(frac, msg):
        print(f"[{frac * 100:5.1f}%] {msg}")

    try:
        m = train_and_save(settings, cli_progress)
    except TrainingError as exc:
        raise SystemExit(f"\nTraining stopped: {exc}")
    except Exception as exc:
        raise SystemExit(f"\nTraining failed: {type(exc).__name__}: {exc}")

    print(f"\nPeople: {m['n_people']}  |  images: {m['n_images']}  |  "
          f"components: {m['n_components_used']}  |  variance kept: {m['explained_variance']:.1%}")
    ev = m["evaluation"]
    if ev:
        print(f"Test accuracy: {ev['accuracy']:.3f}  (with unknown threshold: "
              f"{ev['accuracy_with_threshold']:.3f})")
        print(f"Macro precision/recall/F1: {ev['precision_macro']:.3f} / "
              f"{ev['recall_macro']:.3f} / {ev['f1_macro']:.3f}")
        if ev["unknown_rejection_rate"] is not None:
            print(f"Unseen persons rejected as Unknown: {ev['unknown_rejection_rate']:.1%}")
    print(f"Unknown threshold (auto): {m['auto_threshold']:.3f}")
    for w in m["warnings"]:
        print("WARNING:", w)
    if m["skipped"]:
        print(f"{len(m['skipped'])} image(s) skipped (see models/metrics.json).")
    print(f"Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
