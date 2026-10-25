"""
Recognise every face in a photo using the SAVED PCA model (no retraining).

Usage (PowerShell):
    python train_model.py                       # once, or after changing the dataset
    python predict_face.py "path\\to\\photo.jpg"

The annotated result is saved next to the photo as <name>_result.jpg.
"""

import sys
from pathlib import Path

import cv2

import config
from face_detection import CascadeNotFoundError, FaceDetector
from face_recognition_pca import ModelLoadError, PCAFaceRecognizer, analyse_image
from utils.image_utils import ImageLoadError, load_image_file, resize_max_side

if len(sys.argv) < 2:
    raise SystemExit('Usage: python predict_face.py "path\\to\\photo.jpg"')
photo = Path(sys.argv[1])

settings = config.load_settings()
try:
    model = PCAFaceRecognizer.load(config.MODEL_PATH)
    detector = FaceDetector(settings)
    bgr = resize_max_side(load_image_file(photo), config.MAX_PROCESS_SIDE)
except (ModelLoadError, CascadeNotFoundError, ImageLoadError) as exc:
    raise SystemExit(str(exc))

threshold = config.effective_threshold(settings, model.auto_threshold)
annotated, records = analyse_image(bgr, detector, model, threshold)

if not records:
    raise SystemExit("No face detected. Try a clearer, front-facing image.")
print(f"Unknown threshold: {threshold:.2f}")
for r in records:
    print(f"Face {r['Face']}: {r['Identity']:<15} distance={r['Distance']:.2f} "
          f"match={r['Match score %']:.0f}%  (closest: {r['Closest match']})")

out = photo.with_name(photo.stem + "_result.jpg")
ok, buf = cv2.imencode(".jpg", annotated)
if ok:
    out.write_bytes(buf.tobytes())
    print("Saved", out)
