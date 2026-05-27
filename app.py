import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection",
    page_icon="🧍",
    layout="wide",
)

st.title("🧍 Object Detection with YOLOv8")
st.markdown("Upload an image or video and the model will detect objects in real time.")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

model_path = st.sidebar.text_input(
    "Model path (.pt)",
    value="yolov8n.pt",
    help="Path to your trained YOLOv8 weights file.",
)

confidence = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.10,
    max_value=0.95,
    value=0.25,
    step=0.05,
)

iou_thresh = st.sidebar.slider(
    "IoU threshold (NMS)",
    min_value=0.10,
    max_value=0.95,
    value=0.45,
    step=0.05,
)

show_labels = st.sidebar.checkbox("Show labels", value=True)
show_conf   = st.sidebar.checkbox("Show confidence scores", value=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(path)

try:
    model = load_model(model_path)
    st.sidebar.success(f"✅ Model loaded: `{os.path.basename(model_path)}`")
except FileNotFoundError as e:
    st.sidebar.error(str(e))
    st.stop()

# ── Helper: run inference and annotate ───────────────────────────────────────
def run_detection(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    """Return annotated BGR image and detected object count."""
    results = model.predict(
        source=img_bgr,
        conf=confidence,
        iou=iou_thresh,
        verbose=False,
    )[0]

    annotated = results.plot(labels=show_labels, conf=show_conf)
    count = len(results.boxes) if results.boxes is not None else 0
    return annotated, count

# ── Input tabs ────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["📷 Image", "🎬 Video", "📹 Webcam"])

# ── IMAGE tab ─────────────────────────────────────────────────────────────────
with tab_img:
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    if uploaded:
        pil_img  = Image.open(uploaded).convert("RGB")
        img_bgr  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running detection…"):
            annotated_bgr, count = run_detection(img_bgr)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.subheader(f"Detections — {count} object(s) found")
            st.image(annotated_rgb, use_container_width=True)

        # Download button
        _, buf_col, _ = st.columns([2, 1, 2])
        with buf_col:
            annotated_pil = Image.fromarray(annotated_rgb)
            import io
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download result",
                data=buf.getvalue(),
                file_name="detection_result.png",
                mime="image/png",
            )

# ── VIDEO tab ─────────────────────────────────────────────────────────────────
with tab_vid:
    vid_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="vid_upload",
    )

    if vid_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Video: {total_frames} frames · {fps:.1f} fps")

        frame_placeholder = st.empty()
        metrics_col1, metrics_col2 = st.columns(2)
        objects_metric  = metrics_col1.empty()
        elapsed_metric = metrics_col2.empty()

        stop_btn = st.button("⏹ Stop processing")

        frame_idx   = 0
        total_count = 0
        t0          = time.time()

        progress = st.progress(0)

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_bgr, count = run_detection(frame)
            annotated_rgb        = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(annotated_rgb, use_container_width=True)
            total_count = max(total_count, count)

            elapsed = time.time() - t0
            objects_metric.metric("Objects detected (peak)", total_count)
            elapsed_metric.metric("Elapsed", f"{elapsed:.1f}s")

            frame_idx += 1
            progress.progress(min(frame_idx / max(total_frames, 1), 1.0))

        cap.release()
        os.unlink(tmp_path)
        st.success("✅ Processing complete.")

# ── WEBCAM tab ────────────────────────────────────────────────────────────────
with tab_cam:
    st.info(
        "Click **Start** to capture a single frame from your webcam "
        "and run detection on it."
    )

    if st.button("📸 Capture & detect"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Could not access webcam. Make sure it is connected and not in use.")
        else:
            annotated_bgr, count = run_detection(frame)
            annotated_rgb        = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"{count} object(s) detected", use_container_width=True)
