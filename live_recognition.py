"""
Real-time face recognition in an OpenCV window (replaces predict_webcam.py).
Uses the SAVED model - it does not retrain at start-up.

Run:   python live_recognition.py
Keys (click the video window first):  S = save snapshot to snapshots/   Q = quit

Privacy: snapshots are only written when YOU press S.
"""

import sys
import time

import cv2

import config
from face_detection import CascadeNotFoundError, FaceDetector
from face_recognition_pca import ModelLoadError, PCAFaceRecognizer, analyse_image

settings = config.load_settings()
try:
    model = PCAFaceRecognizer.load(config.MODEL_PATH)
    detector = FaceDetector({**settings, "min_face_size": max(settings["min_face_size"], 80)})
except (ModelLoadError, CascadeNotFoundError) as exc:
    raise SystemExit(str(exc))
threshold = config.effective_threshold(settings, model.auto_threshold)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and try again.")

config.SNAPSHOT_DIR.mkdir(exist_ok=True)
print(f"Webcam running (unknown threshold {threshold:.2f}). S = snapshot, Q = quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera stopped delivering frames.")
        break
    annotated, _ = analyse_image(frame, detector, model, threshold)
    cv2.imshow("Face recognition (S = snapshot, Q = quit)", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        path = config.SNAPSHOT_DIR / f"snapshot_{int(time.time())}.jpg"
        cv2.imwrite(str(path), annotated)
        print("Saved", path)
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
