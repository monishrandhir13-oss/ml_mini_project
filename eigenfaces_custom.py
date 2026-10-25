"""
Eigenfaces on YOUR OWN dataset (PCA + SVM / KNN)

Folder layout:
    dataset/
        PersonA/ img1.jpg img2.jpg ...
        PersonB/ img1.jpg ...

Install: pip install opencv-python numpy scikit-learn matplotlib
Run:     python eigenfaces_custom.py
"""

import os
import cv2
import numpy as np

import config                                   # project paths (no hard-coded D:\\ path)
from face_detection import FaceDetector         # robust Haar cascade loader
from utils.image_utils import load_image_file, ImageLoadError
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------------- SETTINGS ----------------
DATASET_DIR = str(config.DATASET_DIR)          # <project>/dataset
IMG_SIZE = 64
RANDOM_STATE = 42
# ------------------------------------------

face_detector = FaceDetector(config.load_settings())


def extract_face(path):
    """Load an image, crop the largest face, return a 64x64 float array in [0,1]."""
    try:
        img = load_image_file(path)                      # Unicode-safe, EXIF-aware
    except ImageLoadError:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detect(gray)                   # sorted largest first
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]                                # largest face
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = cv2.equalizeHist(face)                         # reduce lighting differences
    return face.astype(np.float32) / 255.0


# ---------------- 1. Load dataset ----------------
images, labels, names = [], [], []
for label, person in enumerate(sorted(os.listdir(DATASET_DIR))):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    names.append(person)
    count = 0
    for fname in os.listdir(person_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        face = extract_face(os.path.join(person_dir, fname))
        if face is None:
            print(f"  [skipped: no face found] {person}/{fname}")
            continue
        images.append(face)
        labels.append(len(names) - 1)
        count += 1
    print(f"{person}: {count} usable faces")

images = np.array(images)
y = np.array(labels)
X = images.reshape(len(images), -1)
print(f"\nTotal: {len(X)} faces, {len(names)} people")

if len(names) < 2 or len(X) < 20:
    raise SystemExit("Need at least 2 people and ~20 usable face images in total.")

# ---------------- 2. Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

cv = min(5, int(np.bincount(y_train).min()))
cv = max(cv, 2)
max_comp = int(len(X_train) * (cv - 1) / cv) - 1
comp_options = [c for c in (5, 10, 20, 30, 50, 100) if c <= max_comp] or [max_comp]

# ---------------- 3. Eigenfaces plot ----------------
n_show = min(15, max_comp)
pca_vis = PCA(n_components=n_show, random_state=RANDOM_STATE).fit(X_train)
fig, axes = plt.subplots(3, 6, figsize=(12, 6))
axes = axes.ravel()
axes[0].imshow(pca_vis.mean_.reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
axes[0].set_title("Mean face")
for i in range(1, len(axes)):
    if i - 1 < n_show:
        axes[i].imshow(pca_vis.components_[i - 1].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        axes[i].set_title(f"Eigenface {i}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("custom_eigenfaces.png", dpi=120)

# ---------------- 4. Train classifiers ----------------
svm_pipe = make_pipeline(PCA(whiten=True, random_state=RANDOM_STATE), SVC(kernel="rbf"))
svm_grid = {
    "pca__n_components": comp_options,
    "svc__C": [1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.001],
}
knn_pipe = make_pipeline(PCA(whiten=True, random_state=RANDOM_STATE), KNeighborsClassifier())
knn_grid = {
    "pca__n_components": comp_options,
    "kneighborsclassifier__n_neighbors": [1, 3],
}

results = {}
for name, pipe, grid in [("SVM", svm_pipe, svm_grid), ("KNN", knn_pipe, knn_grid)]:
    search = GridSearchCV(pipe, grid, cv=cv, n_jobs=-1).fit(X_train, y_train)
    pred = search.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = (search, pred, acc)
    print(f"\n=== PCA + {name} ===")
    print("Best params:", search.best_params_)
    print(f"CV accuracy:   {search.best_score_:.3f}")
    print(f"Test accuracy: {acc:.3f}")

# ---------------- 5. Report + predictions ----------------
best = max(results, key=lambda k: results[k][2])
search, best_pred, best_acc = results[best]
print(f"\nBest model: PCA + {best} ({best_acc:.3f})")
print(classification_report(y_test, best_pred, labels=range(len(names)),
                            target_names=names, zero_division=0))

n = min(12, len(X_test))
fig, axes = plt.subplots(2, 6, figsize=(13, 5))
for ax in axes.ravel():
    ax.axis("off")
for ax, img, t, p in zip(axes.ravel(), X_test[:n], y_test[:n], best_pred[:n]):
    ax.imshow(img.reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
    ax.set_title(f"T:{names[t]}\nP:{names[p]}", fontsize=8,
                 color="green" if t == p else "red")
plt.suptitle(f"Predictions (PCA + {best})")
plt.tight_layout()
plt.savefig("custom_predictions.png", dpi=120)
plt.show()
