# PCA-Based Face Detection and Recognition System

Eigenfaces (PCA) + K-Nearest-Neighbours with unknown-person rejection, OpenCV face detection and a
Streamlit web app. Runs fully offline on Windows.

> **Detection vs recognition.** *Detection* (`face_detection.py`, Haar cascade) finds **where** a
> face is. *Recognition* (`face_recognition_pca.py`, PCA + KNN) decides **who** it is, or says
> `Unknown`.

## 1. Project structure (save every file exactly here)

```
mini_project_ml/                 <- your existing project folder
├── app.py                       Streamlit web app (8 pages)
├── face_detection.py            Haar cascade loading, detection, shared preprocessing, drawing
├── face_recognition_pca.py      PCA + KNN recogniser, unknown rejection, save/load
├── train_model.py               dataset loading, train/test evaluation, model saving (CLI too)
├── config.py                    paths, default settings, admin PIN
├── capture_faces.py             webcam photo collector (fixed)
├── predict_face.py              recognise a photo from the command line (uses saved model)
├── live_recognition.py          real-time OpenCV window (replaces "predict_webcam .py")
├── eigenfaces.py                Olivetti demo (unchanged - it was fine)
├── eigenfaces_custom.py         grid-search experiment on your dataset (paths/cascade fixed)
├── requirements.txt
├── README.md
├── .gitignore
├── utils/  __init__.py  image_utils.py  dataset_utils.py  ui.py
├── dataset/                     one sub-folder per person (created automatically)
├── models/                      pca_face_model.joblib, metrics.json, settings.json
└── snapshots/                   only created if you press S in live_recognition.py
```

Delete the old `predict_webcam .py` (it has a space in its name) and the old `predict_face.py`; the
new files replace them.

## 2. Setup (Windows PowerShell)

Install **Python 3.11** (python.org, tick *Add to PATH*), then:

```powershell
cd C:\path\to\mini_project_ml
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import cv2; print(cv2.__version__); print(cv2.data.haarcascades)"
```

If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry.
The last command must print a 4.x version and an existing folder.

Copy your old photos into the new location (optional):

```powershell
Copy-Item -Recurse "C:\Users\abhis\Downloads\dataset\*" .\dataset\
```

## 3. Train and run

```powershell
python train_model.py          # trains, prints accuracy, saves models\pca_face_model.joblib
streamlit run app.py           # opens http://localhost:8501
```

Optional command-line tools:

```powershell
python capture_faces.py                      # collect webcam photos for one person
python predict_face.py "C:\photos\test.jpg"  # recognise faces in a photo
python live_recognition.py                   # real-time webcam recognition (Q quits)
```

The app loads the saved model at start-up; it retrains only when you press **Train** (or register /
delete people with "retrain" ticked).

Set your own admin PIN before a demo (default is `admin123`):

```powershell
$env:FACE_APP_ADMIN_PIN = "my-secret-pin"
streamlit run app.py
```

## 4. Adding people and photos

**In the app:** *Dataset Management* → enter PIN → *Register / add photos* → type the name, upload
10-25 photos (different angles, lighting, expressions; one face per photo), tick consent → *Save*.
Photos without a face, with several faces, corrupt files and duplicates are rejected with a reason.

**By hand:** create `dataset\<PersonName>\` and copy `.jpg/.png` files into it, then run
`python train_model.py` (or press Train in the app). Names may contain letters, digits, spaces, `-`
and `_`.

**With the webcam:** `python capture_faces.py` (S = save, A = auto-capture, Q = quit).

Tips: at least 2 people; 10+ photos each; the same person must be the only face; front-facing,
good light.

## 5. How the model works / evaluation

1. Detect the largest face, crop, grayscale, resize to 64x64, histogram-equalise, scale to [0,1],
   flatten to 4096 numbers. The **same function** is used for training and prediction.
2. Stratified train/test split (default 75/25).
3. PCA is fitted **on the training part only** (test faces are only projected → no data leakage).
4. KNN (K=3, distance-weighted, Euclidean) classifies in the PCA space.
5. **Unknown rejection:** the distance to the nearest training face of the predicted person must be
   below a threshold, otherwise the label is `Unknown`. Automatic mode calibrates the threshold as
   the 95th percentile of distances of held-out genuine test faces; the Settings page can scale it
   or set it manually.
6. Reported metrics: accuracy, macro precision/recall/F1, confusion matrix, accuracy with the
   threshold, known faces wrongly rejected, and unseen-person rejection rate
   (leave-one-person-out).

## 6. Diagnosis of the original scripts

| # | Problem | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `haarcascade_frontalface_default.xml` not found / `CascadeClassifier` errors | `cv2.data.haarcascades` only exists in OpenCV 4.x pip builds. OpenCV 5.x (what `pip install opencv-python` can install on a new Python) has no `CascadeClassifier` and ships no cascade XML. Also, OpenCV never raises when the XML fails to load - it silently creates an empty classifier. A non-ASCII Windows user-folder name also breaks loading. | `requirements.txt` pins `opencv-python>=4.8,<5`; `face_detection.find_cascade_path()` tries `cv2.data`, the package folder, `models\`, then a download; `.empty()` is checked; ASCII temp-copy fallback; clear error messages. |
| 2 | Hard-coded, **inconsistent** absolute paths (`D:\mini_project_ml\dataset` vs `C:\Users\abhis\Downloads\dataset`) | Each script had its own constant | Everything uses `config.py` (paths relative to the project). |
| 3 | `predict_webcam .py` has a space before `.py` | File naming | Replaced by `live_recognition.py`. |
| 4 | Model retrained from scratch on every prediction / webcam start | No serialization | Model saved with `joblib`, loaded at start-up. |
| 5 | "Unknown" decided by SVM `predict_proba` > 0.6 | SVM probabilities are unreliable on tiny datasets and always sum to 1 across *known* people | Distance threshold in PCA space with calibration. |
| 6 | Final models trained on 100 % of the data, no accuracy shown (`predict_face.py`, `predict_webcam.py`) | No evaluation step | `train_model.py` evaluates on a held-out set first. |
| 7 | `cv2.imread(path)` returns `None` on non-ASCII paths and ignores EXIF rotation | OpenCV limitation | Pillow-based loader reading raw bytes. |
| 8 | Different `minSize` for training (60) and webcam (80), duplicated preprocessing in 4 files | Copy-paste | One `preprocess_face()` + one `FaceDetector`. |
| 9 | `capture_faces.py` could overwrite photos after files were deleted; `.jpeg` not counted | Counter derived from file count | Unique file names, validated person name. |
| 10 | `eigenfaces.py` | Works (Olivetti, PCA inside the CV pipeline - no leakage) | Left unchanged. |

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `AttributeError: module 'cv2' has no attribute 'data'` / `CascadeClassifier` missing / `Can't open haarcascade...` | `pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python` then `pip install "opencv-python>=4.8,<5"`. Only one OpenCV package may be installed. |
| Cascade found but "could not load" | Your user folder has non-ASCII characters or the file is corrupt: the app copies it to a temp folder automatically; otherwise download the XML into `models\haarcascade_frontalface_default.xml`. |
| `ImportError: DLL load failed` (cv2/numpy) | Use Python 3.10-3.12, a fresh venv, and install the *Microsoft Visual C++ Redistributable*. |
| `ModuleNotFoundError` | Venv not activated (`.\.venv\Scripts\Activate.ps1`) or `pip install -r requirements.txt` not run. Run commands from the project folder. |
| Training says "Need at least 2 people with 2+ usable images" | Add more people/photos; check the skipped list on *Train & Evaluate* (faces not detected). |
| `n_components must be between 0 and min(n_samples, n_features)` | Fixed automatically: components are clamped to `n_train - 1`. Add more photos for more eigenfaces. |
| "Dataset too small for a train/test split" | Add 10+ photos per person. |
| Everybody is "Unknown" | Raise the multiplier in Settings (e.g. 1.3) or switch to manual and increase the threshold; add more varied photos; retrain. |
| Strangers are recognised | Lower the multiplier (e.g. 0.8). |
| Wrong person predicted | Photos in the wrong folder, too few/similar photos, very different lighting. |
| Webcam is black / "Could not open the webcam" | Close Zoom/Teams/Camera app; Windows *Settings → Privacy → Camera* → allow desktop apps; try index 1 in `cv2.VideoCapture(1, ...)`. |
| Browser camera widget shows nothing | Click the lock icon in the address bar → allow Camera; use Chrome/Edge on `localhost`; reload. |
| Saved model fails to load after upgrading scikit-learn | Press *Train* again (the app shows the error instead of crashing). |
| `streamlit` is not recognised | Activate the venv, or run `python -m streamlit run app.py`. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| `cv2.imshow` error "The function is not implemented" | You installed `opencv-python-headless`; uninstall it and install `opencv-python`. |

## 8. Presentation / viva: how PCA works here (2 minutes)

* A 64x64 face is a point in a **4096-dimensional** space. Neighbouring pixels are highly
  correlated, so faces really occupy a small *face subspace*.
* **PCA** computes the mean face, subtracts it, and finds the orthogonal directions of greatest
  variance - the eigenvectors of the covariance matrix (computed with SVD). These are the
  **eigenfaces**; the first capture lighting, later ones finer facial structure.
* We keep the top *k* (≈50) eigenfaces. Each face is **projected** to *k* weights: 4096 → 50 numbers,
  keeping ~90 %+ of the variance. A face ≈ mean + Σ wᵢ · eigenfaceᵢ.
* **Recognition:** project the new face the same way, find its nearest training faces by Euclidean
  distance (KNN). If even the nearest one is farther than the threshold, answer *Unknown*.
* **No leakage:** PCA is fitted on the training images only; test images are only projected.
* **Why PCA helps:** noise removal, fewer dimensions → faster and more robust distance comparisons.
* **Limitations:** sensitive to lighting/pose/occlusion; linear method; no liveness check.
  Improvements: LDA (Fisherfaces), LBP/HOG, deep embeddings (FaceNet/ArcFace), face alignment.

Likely viva questions: *Why mean-centre?* (PCA analyses variance around the mean.) *Eigenvalue
meaning?* (variance along that eigenface.) *Why not use all components?* (they mostly encode noise
and increase cost.) *Detection vs recognition?* (above.) *How is Unknown decided?* (distance
threshold.) *How do you choose k?* (cumulative explained variance + test accuracy.)

## 9. Privacy notes

Face images are biometric personal data. They stay in `dataset\` on your own computer; webcam
frames in the app are processed in memory and never written to disk; registering and deleting
people requires the admin PIN and a consent tick; a person's folder can be deleted completely and
the model is then retrained. Do not commit `dataset\` or `models\` to GitHub (the `.gitignore`
excludes them). The system has no liveness detection - do not use it for security decisions.
