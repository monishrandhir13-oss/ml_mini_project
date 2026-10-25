"""
app.py - Streamlit web application: PCA-Based Face Detection and Recognition System.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hmac

import numpy as np
import pandas as pd
import streamlit as st

import config
import face_detection as fd
from face_recognition_pca import ModelLoadError, PCAFaceRecognizer, analyse_image
from train_model import TrainingError, delete_model_files, load_metrics, train_and_save
from utils import dataset_utils as ds
from utils.image_utils import (ImageLoadError, bgr_to_rgb, load_image_file,
                               load_image_from_bytes, resize_max_side)
from utils.ui import card, hero, inject_css, pill

st.set_page_config(page_title="PCA Face Recognition", page_icon="🧑‍💻", layout="wide")
config.ensure_dirs()
inject_css()


# ==========================================================================
# Shared helpers
# ==========================================================================
@st.cache_resource(show_spinner=False)
def _cached_model(mtime: float) -> PCAFaceRecognizer:
    """Load the saved model once; reloaded automatically when the file changes."""
    return PCAFaceRecognizer.load(config.MODEL_PATH)


def get_model():
    """Return (model | None, error_message | None)."""
    if not config.MODEL_PATH.exists():
        return None, None
    try:
        return _cached_model(config.MODEL_PATH.stat().st_mtime), None
    except ModelLoadError as exc:
        return None, str(exc)


def get_detector(settings: dict):
    """Return (FaceDetector | None, error | None)."""
    try:
        return fd.FaceDetector(settings), None
    except fd.CascadeNotFoundError as exc:
        return None, str(exc)


def model_status(model, dataset_changed: bool) -> tuple[str, str]:
    if model is None:
        return "Not trained", "bad"
    return ("Outdated - retrain", "warn") if dataset_changed else ("Ready", "ok")


def dataset_is_outdated(model) -> bool:
    return bool(model) and model.meta.get("dataset_signature") != ds.dataset_signature()


def run_training(settings: dict) -> dict | None:
    """Train with a progress bar; show friendly errors."""
    bar = st.progress(0.0, text="Starting...")

    def cb(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        metrics = train_and_save(settings, cb)
    except fd.CascadeNotFoundError as exc:
        bar.empty(); st.error(str(exc)); return None
    except TrainingError as exc:
        bar.empty(); st.error(str(exc)); return None
    except Exception as exc:                                   # noqa: BLE001
        bar.empty(); st.error(f"Unexpected training error: {type(exc).__name__}: {exc}")
        return None
    bar.empty()
    st.success("Model trained and saved.")
    return metrics


def uploaded_to_bgr(uploaded):
    """Validate an uploaded file and decode it. Returns (bgr, raw_bytes) or raises ImageLoadError."""
    if uploaded.size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageLoadError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")
    raw = uploaded.getvalue()
    return load_image_from_bytes(raw), raw


def show_metrics(metrics: dict) -> None:
    """Render evaluation metrics, confusion matrix and dataset statistics."""
    ev = metrics.get("evaluation")
    c = st.columns(4)
    c[0].metric("People", metrics["n_people"])
    c[1].metric("Face images", metrics["n_images"])
    c[2].metric("PCA components", metrics["n_components_used"],
                f"{metrics['explained_variance']:.1%} variance")
    c[3].metric("Auto threshold", f"{metrics['auto_threshold']:.2f}")

    for w in metrics.get("warnings", []):
        st.warning(w)

    if ev:
        st.subheader("Evaluation on held-out test faces")
        m = st.columns(5)
        m[0].metric("Accuracy", f"{ev['accuracy']:.1%}")
        m[1].metric("Precision (macro)", f"{ev['precision_macro']:.1%}")
        m[2].metric("Recall (macro)", f"{ev['recall_macro']:.1%}")
        m[3].metric("F1 (macro)", f"{ev['f1_macro']:.1%}")
        rej = ev.get("unknown_rejection_rate")
        m[4].metric("Unseen persons rejected", "n/a" if rej is None else f"{rej:.0%}")
        st.caption(
            f"Train: {ev['n_train']} faces | Test: {ev['n_test']} faces (never used to fit PCA "
            f"or KNN). Accuracy with the unknown threshold ({ev['threshold_used']:.2f}): "
            f"{ev['accuracy_with_threshold']:.1%}. Known faces wrongly rejected: "
            f"{ev['known_rejected_rate']:.1%}.")
        names = metrics["names"]
        with st.expander("Confusion matrix (rows = true person, columns = predicted)"):
            st.dataframe(pd.DataFrame(ev["confusion_matrix"], index=names, columns=names))
        with st.expander("Per-person precision / recall / F1"):
            rows = {n: ev["per_person"][n] for n in names if n in ev["per_person"]}
            st.dataframe(pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]])
    else:
        st.info("Accuracy could not be measured because the dataset is too small.")

    st.subheader("Dataset statistics")
    st.bar_chart(pd.Series(metrics["images_per_person"], name="usable faces"))
    st.caption(f"Trained at {metrics['trained_at']} in {metrics['duration_sec']} s.")
    if metrics.get("skipped"):
        with st.expander(f"{len(metrics['skipped'])} image(s) skipped"):
            st.dataframe(pd.DataFrame(metrics["skipped"], columns=["Image", "Reason"]))


def require_model(model, err) -> bool:
    if err:
        st.error(err)
    if model is None:
        st.warning("No trained model yet. Register people on **Dataset Management** and "
                   "press **Train** on the **Train & Evaluate** page (or run "
                   "`python train_model.py`).")
        return False
    return True


# ==========================================================================
# Pages
# ==========================================================================
def page_home(settings):
    hero("PCA-Based Face Detection and Recognition System",
         "Eigenfaces + K-Nearest-Neighbours with unknown-person rejection, built with "
         "OpenCV, scikit-learn and Streamlit.")
    model, err = get_model()
    if err:
        st.error(err)
    summary = ds.dataset_summary()
    outdated = dataset_is_outdated(model)
    status, kind = model_status(model, outdated)
    metrics = load_metrics()
    acc = (metrics or {}).get("evaluation") or {}

    c = st.columns(4)
    with c[0]:
        card("Registered people", str(len(summary)), "folders in dataset/")
    with c[1]:
        card("Training images", str(sum(summary.values())), "photos on disk")
    with c[2]:
        card("Model status", pill(status, kind), "saved model is loaded at start-up")
    with c[3]:
        card("Test accuracy", f"{acc['accuracy']:.0%}" if acc else "-", "held-out faces")

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What is PCA?")
        st.markdown(
            "A face photo of 64 x 64 pixels is a point in a **4096-dimensional** space, "
            "but real faces occupy only a tiny part of it. **Principal Component Analysis** "
            "finds the directions in which faces vary most - the *eigenfaces*. Keeping the "
            "top 50 or so lets us describe each face with ~50 numbers instead of 4096. "
            "Two photos of the same person land close together in this *face space*, so "
            "recognition becomes a simple nearest-neighbour search.")
        st.markdown(
            "**Detection vs recognition.** *Detection* (Haar cascade) finds **where** a face "
            "is. *Recognition* (PCA + KNN) decides **who** it is - or says *Unknown*.")
    with right:
        st.subheader("Quick start")
        st.markdown(
            "1. **Dataset Management** - register people with 10+ photos each\n"
            "2. **Train & Evaluate** - train and check the accuracy\n"
            "3. **Upload & Recognize** or **Live Detection** - try it out")
        if outdated:
            st.warning("The dataset changed since the last training - retrain the model.")
    det, derr = get_detector(settings)
    if derr:
        st.error(derr)


def page_live(settings):
    hero("Live Face Detection", "Capture frames from your webcam and detect faces.")
    det, derr = get_detector(settings)
    if derr:
        st.error(derr); return
    model, err = get_model()
    if err:
        st.error(err)

    st.info("**Privacy:** captured frames are processed in memory and are **not saved** "
            "anywhere. Your browser asks for camera permission; nothing leaves this computer "
            "when you run the app locally.")
    st.caption("Streamlit's camera widget works with captured frames (snapshot mode). "
               "For a true real-time video window run `python live_recognition.py`.")

    recognise = st.checkbox("Also recognise people", value=False, disabled=model is None,
                            help="Needs a trained model.")
    if "cam_key" not in st.session_state:
        st.session_state.cam_key = 0
    on = st.toggle("Camera on (start / stop)", value=False)
    if not on:
        st.caption("Switch the camera on to begin. Switching it off releases the camera.")
        return

    c1, _ = st.columns([1, 3])
    if c1.button("Retake / clear"):
        st.session_state.cam_key += 1
        st.rerun()
    try:
        shot = st.camera_input("Press **Take photo** to capture and analyse a frame",
                               key=f"cam_{st.session_state.cam_key}")
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Camera error: {exc}"); return
    if shot is None:
        st.caption("No camera access? Click the camera/lock icon in the browser address bar, "
                   "allow the camera, close other apps that use it (Zoom, Teams) and reload.")
        return
    try:
        bgr = resize_max_side(load_image_from_bytes(shot.getvalue()), config.MAX_PROCESS_SIDE)
        thr = config.effective_threshold(settings, model.auto_threshold if model else None)
        out, rec = analyse_image(bgr, det, model if (recognise and model) else None, thr)
    except Exception as exc:                                    # noqa: BLE001
        st.error(f"Could not process the frame: {exc}"); return
    st.image(bgr_to_rgb(out), caption=f"{len(rec)} face(s) detected", use_container_width=True)
    if rec:
        st.dataframe(pd.DataFrame(rec), hide_index=True)
    else:
        st.warning("No face found. Face the camera, improve the lighting, move closer.")


def page_upload(settings):
    hero("Image Upload and Recognition", "Upload photos - every face is detected and identified.")
    det, derr = get_detector(settings)
    if derr:
        st.error(derr); return
    model, err = get_model()
    if not require_model(model, err):
        st.caption("You can still use **Live Detection** to test face detection only.")
        return
    if model.image_size != settings["image_size"]:
        st.warning(f"Settings say {settings['image_size']} px but the model was trained with "
                   f"{model.image_size} px. The model's size is used; retrain to apply changes.")
    thr = config.effective_threshold(settings, model.auto_threshold)
    st.caption(f"Unknown threshold: **{thr:.2f}** (distance in PCA space; "
               f"{'auto' if settings['auto_threshold'] else 'manual'}). Change it in Settings.")

    files = st.file_uploader("Upload JPG / JPEG / PNG images", type=config.UPLOAD_TYPES,
                             accept_multiple_files=True)
    for up in files or []:
        st.markdown(f"##### {up.name}")
        try:
            bgr, _ = uploaded_to_bgr(up)
            bgr = resize_max_side(bgr, config.MAX_PROCESS_SIDE)
            out, rec = analyse_image(bgr, det, model, thr)
        except ImageLoadError as exc:
            st.error(f"{up.name}: {exc}"); continue
        except Exception as exc:                                # noqa: BLE001
            st.error(f"{up.name}: processing failed ({exc})"); continue
        a, b = st.columns(2)
        a.image(bgr_to_rgb(bgr), caption="Original", use_container_width=True)
        b.image(bgr_to_rgb(out), caption="Detected + recognised", use_container_width=True)
        if rec:
            st.dataframe(pd.DataFrame(rec), hide_index=True)
            st.caption("Distance: smaller = more similar. 'Match score' is a heuristic, "
                       "not a probability.")
        else:
            st.warning("No face detected in this image.")


def _admin_gate() -> bool:
    if st.session_state.get("is_admin"):
        return True
    st.info("Registering or deleting face data is restricted. Enter the admin PIN.")
    if config.using_default_pin():
        st.warning("Default PIN in use. Set your own before a demo/deployment: "
                   "PowerShell `$env:FACE_APP_ADMIN_PIN=\"your-pin\"` then start the app.")
    with st.form("pin_form"):
        pin = st.text_input("Admin PIN", type="password")
        if st.form_submit_button("Unlock"):
            if hmac.compare_digest(pin.encode(), config.admin_pin().encode()):
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Wrong PIN.")
    return False


def page_dataset(settings):
    hero("Dataset Management", "Register people, add photos, remove data, retrain.")
    if not _admin_gate():
        return
    det, derr = get_detector(settings)
    if derr:
        st.error(derr); return
    if st.sidebar.button("Lock admin area"):
        st.session_state.is_admin = False
        st.rerun()

    tab_add, tab_view, tab_del = st.tabs(["➕ Register / add photos", "📂 Browse", "🗑️ Remove"])

    with tab_add:
        with st.form("register", clear_on_submit=False):
            name = st.text_input("Person's name", max_chars=40,
                                 help="Letters, digits, spaces, - and _. Use an existing name to add photos.")
            files = st.file_uploader(
                f"Face photos (JPG / PNG, {config.RECOMMENDED_IMAGES_PER_PERSON}+ recommended, "
                "one clearly visible face each)", type=config.UPLOAD_TYPES,
                accept_multiple_files=True)
            consent = st.checkbox("I confirm this person has given consent for their photos "
                                  "to be stored and used for recognition.")
            auto_train = st.checkbox("Retrain the model after saving", value=True)
            submitted = st.form_submit_button("Save photos", type="primary")
        if submitted:
            clean, problem = ds.validate_person_name(name)
            if problem:
                st.error(problem)
            elif not files:
                st.error("Please choose at least one photo.")
            elif not consent:
                st.error("Please confirm consent first.")
            else:
                saved, rejected = 0, []
                for up in files:
                    try:
                        bgr, raw = uploaded_to_bgr(up)
                        gray = fd.to_gray(resize_max_side(bgr, config.MAX_PROCESS_SIDE))
                        n_faces = len(det.detect(gray))
                        if n_faces == 0:
                            rejected.append((up.name, "no face detected")); continue
                        if n_faces > 1:
                            rejected.append((up.name, f"{n_faces} faces found - use one face per photo")); continue
                        status, _ = ds.save_face_image(clean, bgr, raw)
                        if status == "saved":
                            saved += 1
                        else:
                            rejected.append((up.name, f"duplicate (already stored for {status.split(':', 1)[1]})"))
                    except ImageLoadError as exc:
                        rejected.append((up.name, str(exc)))
                    except Exception as exc:                    # noqa: BLE001
                        rejected.append((up.name, f"error: {exc}"))
                if saved:
                    st.success(f"Saved {saved} photo(s) for **{clean}** "
                               f"(total {len(ds.list_images(clean))}).")
                if rejected:
                    st.warning(f"{len(rejected)} photo(s) not saved:")
                    st.dataframe(pd.DataFrame(rejected, columns=["File", "Reason"]), hide_index=True)
                if saved and auto_train:
                    run_training(settings)
                elif saved:
                    st.info("Remember to retrain the model on the Train & Evaluate page.")

    with tab_view:
        summary = ds.dataset_summary()
        if not summary:
            st.info("The dataset is empty.")
        else:
            st.dataframe(pd.DataFrame({"Person": list(summary), "Images": list(summary.values())}),
                         hide_index=True)
            who = st.selectbox("Preview photos of", list(summary))
            imgs = ds.list_images(who)[:12]
            cols = st.columns(6)
            for i, path in enumerate(imgs):
                try:
                    cols[i % 6].image(bgr_to_rgb(load_image_file(path)), use_container_width=True)
                except ImageLoadError:
                    cols[i % 6].caption("unreadable")

    with tab_del:
        summary = ds.dataset_summary()
        if not summary:
            st.info("Nothing to remove.")
        else:
            who = st.selectbox("Person to remove", list(summary), key="del_who")
            st.warning(f"This permanently deletes **{summary[who]}** photo(s) of **{who}**.")
            typed = st.text_input("Type the person's name to confirm")
            if st.button("Delete this person", type="primary"):
                if typed.strip() != who:
                    st.error("The typed name does not match.")
                else:
                    removed = ds.delete_person(who)
                    st.success(f"Deleted {removed} photo(s) of {who}.")
                    if len(ds.list_people()) >= 2:
                        run_training(settings)
                    else:
                        delete_model_files()
                        st.info("Fewer than 2 people remain, so the trained model was removed.")


def page_train(settings):
    hero("Model Training and Evaluation", "Train PCA + KNN, then inspect honest test metrics.")
    summary = ds.dataset_summary()
    model, err = get_model()
    if err:
        st.error(err)
    if not summary:
        st.warning("The dataset is empty. Register people on **Dataset Management** first.")
    else:
        st.write(f"**{len(summary)}** people, **{sum(summary.values())}** photos in the dataset.")
        thin = [p for p, n in summary.items() if n < config.RECOMMENDED_IMAGES_PER_PERSON]
        if thin:
            st.caption("Fewer than 10 photos (accuracy may suffer): " + ", ".join(thin))
    if dataset_is_outdated(model):
        st.warning("The dataset changed since the model was trained. Retrain.")
    if st.button("🚀 Train / retrain model", type="primary", disabled=not summary):
        metrics = run_training(settings)
        if metrics:
            show_metrics(metrics)
        return
    metrics = load_metrics()
    if model is not None and metrics:
        st.success("A saved model is loaded - no retraining needed after restarting the app.")
        show_metrics(metrics)
        with st.expander("Danger zone"):
            if st.button("Delete the trained model files"):
                delete_model_files()
                st.rerun()
    elif model is None and summary:
        st.info("No model yet - press the button above.")


def page_settings(settings):
    hero("Settings", "Image size, PCA components, KNN and recognition threshold.")
    det, derr = get_detector(settings)
    st.caption("Changing image size, components or neighbours needs **retraining**. "
               "Threshold and detector settings apply immediately.")
    with st.form("settings"):
        c1, c2 = st.columns(2)
        with c1:
            sizes = [32, 48, 64, 80, 96, 128]
            size = st.select_slider("Face image size (px)", sizes,
                                    value=settings["image_size"] if settings["image_size"] in sizes else 64)
            comps = st.slider("PCA components (eigenfaces)", 2, 200, settings["n_components"])
            knn = st.slider("KNN neighbours (K)", 1, 9, settings["n_neighbors"])
            test = st.slider("Test split", 0.1, 0.5, float(settings["test_size"]), 0.05)
        with c2:
            auto = st.radio("Unknown threshold mode",
                            ["Automatic (calibrated in training)", "Manual"],
                            index=0 if settings["auto_threshold"] else 1)
            scale = st.slider("Automatic threshold multiplier", 0.3, 3.0,
                              float(settings["threshold_scale"]), 0.05,
                              help="Below 1 = stricter (more Unknown); above 1 = more lenient.")
            manual = st.number_input("Manual threshold (distance)", 0.05, 100.0,
                                     float(settings["manual_threshold"]), 0.1)
        st.markdown("**Face detector (Haar cascade)**")
        d1, d2, d3 = st.columns(3)
        sf = d1.slider("Scale factor", 1.02, 1.5, float(settings["scale_factor"]), 0.01)
        mn = d2.slider("Min neighbours", 1, 12, settings["min_neighbors"])
        ms = d3.slider("Min face size (px)", 20, 300, settings["min_face_size"])
        saved = st.form_submit_button("Save settings", type="primary")
    if saved:
        new = config.save_settings({
            **settings, "image_size": size, "n_components": comps, "n_neighbors": knn,
            "test_size": test, "auto_threshold": auto.startswith("Automatic"),
            "threshold_scale": scale, "manual_threshold": manual,
            "scale_factor": sf, "min_neighbors": mn, "min_face_size": ms})
        st.success("Settings saved.")
        if any(new[k] != settings[k] for k in ("image_size", "n_components", "n_neighbors", "test_size")):
            st.info("Retrain the model to apply the changed training settings.")
    if st.button("Reset to defaults"):
        config.save_settings(config.DEFAULT_SETTINGS)
        st.rerun()
    st.divider()
    if derr:
        st.error(derr)
    else:
        st.caption(f"Haar cascade in use: `{fd.find_cascade_path()}`")


def page_eigenfaces(settings):
    hero("Eigenfaces and PCA Explained", "How faces become a handful of numbers.")
    st.markdown(
        "**Step by step**\n\n"
        "1. Every training face becomes a vector of 64 x 64 = 4096 pixel values.\n"
        "2. Subtract the **mean face** (the average of all faces).\n"
        "3. PCA finds orthogonal directions of largest variance - the **eigenfaces**. "
        "They are eigenvectors of the covariance matrix of the training faces.\n"
        "4. Keep the top *k*. Every face is now described by *k* weights "
        "(its projection on each eigenface).\n"
        "5. A new face is projected the same way and compared with training faces by "
        "Euclidean distance (KNN). Too far from everyone → **Unknown**.\n\n"
        "A face ≈ mean face + w₁·eigenface₁ + w₂·eigenface₂ + ... + w_k·eigenface_k")
    model, err = get_model()
    if err:
        st.error(err)
    if model is None:
        st.info("Train a model to see the real eigenfaces of your dataset.")
        return
    st.subheader("Mean face and top eigenfaces")
    imgs = [model.to_uint8(model.mean_face())] + [model.to_uint8(e) for e in model.eigenfaces(11)]
    caps = ["Mean face"] + [f"Eigenface {i}" for i in range(1, len(imgs))]
    cols = st.columns(6)
    for i, (im, cap) in enumerate(zip(imgs, caps)):
        cols[i % 6].image(im, caption=cap, use_container_width=True, clamp=True)
    st.subheader("How much information do the components keep?")
    cum = np.cumsum(model.pca.explained_variance_ratio_)
    st.line_chart(pd.DataFrame({"Cumulative explained variance": cum}, index=np.arange(1, len(cum) + 1)))

    st.subheader("Reconstruction demo")
    people = [p for p in ds.list_people() if ds.list_images(p)]
    det, derr = get_detector(settings)
    if not people or derr:
        st.caption("Needs dataset photos and a working face detector.")
        return
    who = st.selectbox("Person", people)
    k = st.slider("Number of eigenfaces used", 1, len(model.pca.components_),
                  min(10, len(model.pca.components_)))
    try:
        bgr = resize_max_side(load_image_file(ds.list_images(who)[0]), config.MAX_PROCESS_SIDE)
        vec, _ = fd.extract_largest_face(bgr, det, model.image_size)
    except ImageLoadError:
        vec = None
    if vec is None:
        st.caption("No usable face found in the first photo of this person.")
        return
    a, b = st.columns(2)
    side = model.image_size
    a.image(model.to_uint8(vec.reshape(side, side)), caption="Original (preprocessed)",
            use_container_width=True, clamp=True)
    b.image(model.to_uint8(model.reconstruct(vec, k)), caption=f"Reconstructed from {k} eigenfaces",
            use_container_width=True, clamp=True)


def page_info(settings):
    hero("Project Information", "Face Detection and Recognition Using Principal Component Analysis")
    st.markdown("""
**Problem statement** - Identify people from face images automatically, with a method that is
simple, fast and explainable, and that refuses to name people it has never seen.

**Objectives**
- Detect faces in images and webcam frames.
- Represent faces compactly with PCA (eigenfaces) and recognise them with KNN.
- Reject unknown people with a calibrated distance threshold.
- Evaluate honestly on unseen test images and provide a usable web interface.

**Methodology**
1. Collect photos per person → 2. Haar-cascade detection → 3. crop, grayscale, resize,
histogram-equalise, flatten → 4. stratified train/test split → 5. fit PCA on training faces only →
6. project → 7. KNN in face space → 8. threshold on the nearest distance → 9. metrics.

**Algorithms** - Viola-Jones Haar cascade (detection); PCA / eigenfaces via SVD (feature
extraction); K-Nearest Neighbours, distance-weighted (classification).

**Advantages** - Fast, small model, no GPU, easy to explain, works with few images per person,
eigenfaces are visual and great for teaching.

**Limitations** - Sensitive to lighting, pose, expression and occlusion; Haar cascades miss
side/tilted faces; PCA is linear and global; no liveness check, so a printed photo can fool it;
accuracy drops with many similar-looking people.

**Future scope** - Face alignment, LBP/HOG features, LDA (Fisherfaces), SVM or deep embeddings
(FaceNet/ArcFace), DNN detectors, liveness detection, real-time video with streamlit-webrtc.

**Privacy** - Face data is biometric personal data. Photos stay in `dataset/` on this computer,
webcam frames are never saved by the app, the dataset area is PIN-protected, registration needs
consent, and any person can be deleted completely.
""")


# ==========================================================================
# Navigation
# ==========================================================================
PAGES = {
    "🏠 Home": page_home,
    "🎥 Live Detection": page_live,
    "🖼️ Upload & Recognize": page_upload,
    "🗂️ Dataset Management": page_dataset,
    "🧠 Train & Evaluate": page_train,
    "⚙️ Settings": page_settings,
    "📚 Eigenfaces & PCA": page_eigenfaces,
    "ℹ️ Project Info": page_info,
}

with st.sidebar:
    st.markdown("### 🧑‍💻 PCA Face Recognition")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    _model, _ = get_model()
    _status, _kind = model_status(_model, dataset_is_outdated(_model))
    st.markdown(f"Model: {pill(_status, _kind)}", unsafe_allow_html=True)
    st.caption(f"{len(ds.list_people())} people · {sum(ds.dataset_summary().values())} photos")

PAGES[choice](config.load_settings())
