"""
Face Recognition with Eigenfaces (PCA + SVM / KNN)
Dataset: Olivetti faces (400 images, 40 people, 10 each, 64x64 grayscale)

Install: pip install scikit-learn matplotlib numpy
Run:     python eigenfaces.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
faces = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_STATE)
X, y, images = faces.data, faces.target, faces.images
h, w = images.shape[1:]
print(f"Images: {X.shape[0]} | Pixels/image: {X.shape[1]} | People: {len(np.unique(y))}")

# Stratified split so every person appears in both train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------
# 2. Fit PCA on TRAIN data only (avoids leakage)
#    Olivetti pixels are already scaled to [0, 1], so we only
#    mean-center (PCA does this internally).
# ---------------------------------------------------------------
pca_full = PCA(n_components=150, whiten=True, random_state=RANDOM_STATE).fit(X_train)

# ---------------------------------------------------------------
# 3. Visualize: mean face + top eigenfaces
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(11, 7))
axes[0, 0].imshow(pca_full.mean_.reshape(h, w), cmap="gray")
axes[0, 0].set_title("Mean face")
for i, ax in enumerate(axes.ravel()[1:]):
    ax.imshow(pca_full.components_[i].reshape(h, w), cmap="gray")
    ax.set_title(f"Eigenface {i + 1}")
for ax in axes.ravel():
    ax.axis("off")
plt.suptitle("Mean face and top eigenfaces")
plt.tight_layout()
plt.savefig("eigenfaces.png", dpi=120)

# ---------------------------------------------------------------
# 4. Explained variance (scree / cumulative)
# ---------------------------------------------------------------
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = int(np.argmax(cum_var >= 0.95)) + 1 if cum_var[-1] >= 0.95 else None
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_var) + 1), cum_var)
plt.axhline(0.95, color="r", ls="--", label="95% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.legend()
plt.tight_layout()
plt.savefig("explained_variance.png", dpi=120)
print(f"Components for 95% variance: {n_95}")

# ---------------------------------------------------------------
# 5. Reconstruction quality vs. number of components
# ---------------------------------------------------------------
sample = X_test[0:1]
ks = [10, 30, 60, 100, 150]
fig, axes = plt.subplots(1, len(ks) + 1, figsize=(12, 2.5))
axes[0].imshow(sample.reshape(h, w), cmap="gray")
axes[0].set_title("Original")
for ax, k in zip(axes[1:], ks):
    p = PCA(n_components=k, random_state=RANDOM_STATE).fit(X_train)
    recon = p.inverse_transform(p.transform(sample))
    ax.imshow(recon.reshape(h, w), cmap="gray")
    ax.set_title(f"k={k}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("reconstruction.png", dpi=120)

# ---------------------------------------------------------------
# 6. Classifiers: PCA -> SVM and PCA -> KNN, tuned with GridSearchCV
#    PCA lives inside the pipeline, so each CV fold refits it
#    on that fold's training data only.
# ---------------------------------------------------------------
svm_pipe = make_pipeline(
    PCA(whiten=True, random_state=RANDOM_STATE), SVC(kernel="rbf")
)
svm_grid = {
    "pca__n_components": [30, 50, 100],
    "svc__C": [1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.001],
}

knn_pipe = make_pipeline(PCA(whiten=True, random_state=RANDOM_STATE), KNeighborsClassifier())
knn_grid = {
    "pca__n_components": [30, 50, 100],
    "kneighborsclassifier__n_neighbors": [1, 3, 5],
}

results = {}
for name, pipe, grid in [("SVM", svm_pipe, svm_grid), ("KNN", knn_pipe, knn_grid)]:
    search = GridSearchCV(pipe, grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = (search, y_pred, acc)
    print(f"\n=== PCA + {name} ===")
    print("Best params:", search.best_params_)
    print(f"CV accuracy:   {search.best_score_:.3f}")
    print(f"Test accuracy: {acc:.3f}")

# Baseline: SVM on raw pixels (does PCA help?)
raw_svm = SVC(kernel="rbf", C=10, gamma="scale").fit(X_train, y_train)
print(f"\nBaseline SVM on raw pixels, test accuracy: "
      f"{accuracy_score(y_test, raw_svm.predict(X_test)):.3f}")

# ---------------------------------------------------------------
# 7. Detailed report + sample predictions for the best model
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k][2])
_, best_pred, best_acc = results[best_name]
print(f"\nBest model: PCA + {best_name} ({best_acc:.3f})")
print(classification_report(y_test, best_pred, zero_division=0))

fig, axes = plt.subplots(2, 6, figsize=(12, 5))
for ax, img, true, pred in zip(axes.ravel(), X_test[:12], y_test[:12], best_pred[:12]):
    ax.imshow(img.reshape(h, w), cmap="gray")
    ax.set_title(f"T:{true} P:{pred}", color="green" if true == pred else "red")
    ax.axis("off")
plt.suptitle(f"Predictions (PCA + {best_name})")
plt.tight_layout()
plt.savefig("predictions.png", dpi=120)

plt.show()
