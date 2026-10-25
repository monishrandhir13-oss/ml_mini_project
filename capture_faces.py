"""
Capture face photos from your webcam into dataset/<PersonName>/

Run (from the project folder):   python capture_faces.py

Controls (click the video window first):
    S = save one photo
    A = toggle auto-capture (saves a photo every 0.6 s while a face is visible)
    Q = quit

Fixes vs. the original: relative dataset path (config.DATASET_DIR), robust Haar
cascade loading, validated person name, never overwrites existing photos,
CAP_DSHOW only on Windows.
"""

import sys
import time
import uuid

import cv2

import config
from face_detection import CascadeNotFoundError, FaceDetector, draw_label, to_gray
from utils import dataset_utils as ds

TARGET_PHOTOS = 25

name, problem = ds.validate_person_name(input("Enter person's name: "))
if problem:
    raise SystemExit(problem)

person_dir = ds.person_path(name)
person_dir.mkdir(parents=True, exist_ok=True)
count = len(ds.list_images(name))

try:
    detector = FaceDetector({**config.load_settings(), "min_face_size": 80})
except CascadeNotFoundError as exc:
    raise SystemExit(str(exc))

if sys.platform.startswith("win"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # DirectShow works best on Windows
else:
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open the webcam. Close other apps using it and try again.")

auto = False
last_save = 0.0


def save(frame):
    """Save the clean frame under a unique name (cannot overwrite older photos)."""
    global count, last_save
    path = person_dir / f"cam_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    if cv2.imwrite(str(path), frame):
        count += 1
        last_save = time.time()
    else:
        print("Could not write", path)


while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera stopped delivering frames.")
        break

    clean = frame.copy()                          # save the frame without drawings
    faces = detector.detect(to_gray(frame))
    for box in faces:
        draw_label(frame, box, "", (0, 255, 0))

    if auto and faces and time.time() - last_save > 0.6:
        save(clean)

    status = f"{name}: {count}/{TARGET_PHOTOS} | auto: {'ON' if auto else 'OFF'} | S=save A=auto Q=quit"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow("Capture faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        if faces:
            save(clean)
        else:
            print("No face detected - photo not saved.")
    elif key == ord("a"):
        auto = not auto
    elif key == ord("q") or count >= TARGET_PHOTOS:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. {count} photos in {person_dir}")
print("Next: python train_model.py")
