import math
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="Aircraft Vision",
    layout="wide",
    initial_sidebar_state="expanded",
)


BOX_COLOR = (20, 255, 57)
MASK_COLOR = (255, 200, 0)
NOSE_COLOR = (0, 0, 255)
TAIL_COLOR = (0, 255, 255)
ARROW_COLOR = (10, 10, 10)


st.markdown(
    """
    <style>
      .block-container {
          padding-top: 2.5rem;
          padding-bottom: 3rem;
          max-width: 1280px;
      }

      h1 {
          font-weight: 600;
          font-size: 2rem;
          letter-spacing: -0.02em;
          margin-bottom: 0.15rem;
      }

      .wordmark {
          font-size: 0.72rem;
          letter-spacing: 0.22em;
          text-transform: uppercase;
          color: #E8A33D;
          font-weight: 600;
          margin-bottom: 0.4rem;
      }

      .caption-line {
          color: #8B8F98;
          font-size: 0.92rem;
          margin-bottom: 1.6rem;
      }

      .section-label {
          font-size: 0.72rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: #8B8F98;
          font-weight: 600;
          margin-bottom: 0.6rem;
      }

      .metric-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1rem;
          margin-top: 1.6rem;
      }

      .metric-card {
          border: 1px solid rgba(255, 255, 255, 0.07);
          background: rgba(255, 255, 255, 0.015);
          border-radius: 8px;
          padding: 1rem 1.15rem;
      }

      .metric-label {
          font-size: 0.7rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: #8B8F98;
          font-weight: 600;
          margin-bottom: 0.35rem;
      }

      .metric-value {
          font-size: 1.55rem;
          font-weight: 600;
          color: #E6E7EA;
          font-variant-numeric: tabular-nums;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      }

      .empty-state {
          border: 1px dashed rgba(255, 255, 255, 0.12);
          border-radius: 10px;
          padding: 3.5rem 2rem;
          text-align: center;
          color: #8B8F98;
          background: rgba(255, 255, 255, 0.012);
          margin-top: 1.5rem;
      }

      .empty-state strong {
          color: #E6E7EA;
          display: block;
          font-size: 1.05rem;
          margin-bottom: 0.4rem;
          font-weight: 500;
      }

      section[data-testid="stSidebar"] {
          border-right: 1px solid rgba(255, 255, 255, 0.06);
      }

      section[data-testid="stSidebar"] .block-container {
          padding-top: 1.5rem;
          padding-bottom: 1.5rem;
      }

      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
          gap: 0.5rem !important;
      }

      section[data-testid="stSidebar"] h3 {
          font-size: 0.85rem;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          color: #E6E7EA;
          font-weight: 700;
          margin: 2rem 0 0.5rem 0;
          padding: 0;
      }

      section[data-testid="stSidebar"] h3:first-child {
          margin-top: 0;
      }

      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
          margin: 0 0 0.5rem 0 !important;
      }

      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
          font-size: 0.78rem;
          color: #7B7F88;
          line-height: 1.4;
          margin: 0;
      }

      section[data-testid="stSidebar"] .stFileUploader {
          margin-bottom: 0.4rem;
      }

      section[data-testid="stSidebar"] .stSelectbox {
          margin-bottom: 0.4rem;
      }

      section[data-testid="stSidebar"] .stSlider {
          margin-bottom: 0.4rem;
      }

      section[data-testid="stSidebar"] .stCheckbox {
          margin-bottom: -0.3rem;
      }

      section[data-testid="stSidebar"] [data-testid="stExpander"] {
          margin-top: 0.75rem;
      }

      .stButton > button {
          border-radius: 6px;
          border: 1px solid rgba(255, 255, 255, 0.12);
          font-weight: 500;
          letter-spacing: 0.02em;
          transition: all 0.15s ease;
      }

      .stButton > button[kind="primary"] {
          background: #E8A33D;
          color: #0E0F12;
          border: 1px solid #E8A33D;
      }

      .stButton > button[kind="primary"]:hover {
          background: #F0B255;
          border-color: #F0B255;
      }

      .stImage img {
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.06);
      }

      hr {
          margin: 2rem 0;
          opacity: 0.08;
      }

      #MainMenu, footer {
          visibility: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return YOLO("models/segmentation.pt"), YOLO("models/pose.pt")


segmentation_model, pose_model = load_models()


EXAMPLES = {
    "Example 1": "app_assets/examples/example1.png",
    "Example 2": "app_assets/examples/example2.png",
    "Example 3": "app_assets/examples/example3.png",
}


st.markdown('<div class="wordmark">Aircraft Vision</div>', unsafe_allow_html=True)
st.markdown("# Aerial Detection & Orientation")
st.markdown(
    '<div class="caption-line">'
    'Instance segmentation and orientation estimation for aerial imagery.'
    '</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### Input")
    st.caption("Upload an aerial image, or pick one of the examples below.")

    uploaded_file = st.file_uploader(
        "Upload your own image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    selected_example = st.selectbox("Or try an example", ["None", *EXAMPLES])

    st.markdown("### Overlay")
    st.caption("Choose what to draw on the result image.")

    col_a, col_b = st.columns(2)
    with col_a:
        show_boxes = st.checkbox(
            "Boxes", value=True,
            help="Rectangle around each detected aircraft.",
        )
        show_orientation = st.checkbox(
            "Arrow", value=True,
            help="Heading direction, drawn from tail to nose.",
        )
    with col_b:
        show_masks = st.checkbox(
            "Masks", value=True,
            help="Pixel-level silhouette of each aircraft.",
        )
        show_keypoints = st.checkbox(
            "Points", value=True,
            help="Nose (red) and tail (yellow) keypoints.",
        )

    st.markdown("### Detection")
    st.caption("Tune how confident the model must be to report a detection.")

    confidence = st.slider(
        "Confidence threshold",
        0.10, 0.90, 0.40, 0.05,
        help="Higher values make the model more conservative.",
    )

    st.caption(f"Showing detections above **{confidence:.2f}** confidence.")

    with st.expander("About this model"):
        st.markdown(
            "Trained on **aerial and satellite imagery**. Performance may "
            "degrade on oblique, ground-level, or low-resolution images."
        )


image = None
source_label = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    source_label = "Uploaded image"
elif selected_example != "None":
    path = EXAMPLES[selected_example]
    if not os.path.exists(path):
        st.error(f"Could not find {path}")
        st.stop()
    image = Image.open(path).convert("RGB")
    source_label = selected_example


if image is None:
    st.markdown(
        '<div class="empty-state">'
        '<strong>No image selected</strong>'
        'Upload an image or pick an example from the sidebar to begin.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


left, right = st.columns(2, gap="large")

with left:
    st.markdown(
        f'<div class="section-label">Input · {source_label}</div>',
        unsafe_allow_html=True,
    )
    st.image(image, width="stretch")
    analyze = st.button("Analyze image", type="primary", width="stretch")


if not analyze:
    with right:
        st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="empty-state">'
            '<strong>Awaiting analysis</strong>'
            'Detection, segmentation, and orientation will appear here.'
            '</div>',
            unsafe_allow_html=True,
        )
    st.stop()


with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
    image.save(tmp.name)
    image_path = tmp.name

start = time.time()
seg_results = segmentation_model.predict(
    source=image_path, conf=confidence, verbose=False
)
pose_results = pose_model.predict(
    source=image_path, conf=confidence, verbose=False
)
elapsed = time.time() - start


output = np.array(image).copy()
seg_result = seg_results[0]

detected_boxes = []

if seg_result.boxes is not None:
    for box in seg_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        detected_boxes.append((x1, y1, x2, y2))
        if show_boxes:
            cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)

if show_masks and seg_result.masks is not None:
    h, w = output.shape[:2]
    accent = np.array(MASK_COLOR, dtype=np.uint8)

    for mask in seg_result.masks.data.cpu().numpy():
        mask = cv2.resize(mask, (w, h)) > 0.5
        overlay = output.copy()
        overlay[mask] = (0.85 * overlay[mask] + 0.15 * accent).astype(np.uint8)
        output = overlay


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )
    return 0 if union == 0 else inter / union


pose_result = pose_results[0]
orientation_count = 0

if (
    pose_result.keypoints is not None
    and pose_result.boxes is not None
    and detected_boxes
):
    keypoints = pose_result.keypoints.xy.cpu().numpy()
    pose_boxes = pose_result.boxes.xyxy.cpu().numpy()

    for points, pose_box in zip(keypoints, pose_boxes):
        if len(points) < 2:
            continue

        pose_box = tuple(map(int, pose_box))
        best_iou = max((iou(pose_box, db) for db in detected_boxes), default=0)
        if best_iou < 0.50:
            continue

        nose_x, nose_y = map(int, points[0])
        tail_x, tail_y = map(int, points[1])

        if min(nose_x, nose_y, tail_x, tail_y) <= 0:
            continue

        orientation_count += 1

        if show_keypoints:
            for (px, py), color in (
                ((tail_x, tail_y), TAIL_COLOR),
                ((nose_x, nose_y), NOSE_COLOR),
            ):
                cv2.circle(output, (px, py), 10, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(output, (px, py), 7, color, -1, cv2.LINE_AA)

        if show_orientation:
            dx, dy = nose_x - tail_x, nose_y - tail_y
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            scale = 1.8
            end = (int(tail_x + dx * scale), int(tail_y + dy * scale))

            cv2.arrowedLine(
                output, (tail_x, tail_y), end, (255, 255, 255), 6,
                tipLength=0.18, line_type=cv2.LINE_AA,
            )
            cv2.arrowedLine(
                output, (tail_x, tail_y), end, ARROW_COLOR, 3,
                tipLength=0.18, line_type=cv2.LINE_AA,
            )


os.remove(image_path)


with right:
    st.markdown(
        '<div class="section-label">Result · Overlay</div>',
        unsafe_allow_html=True,
    )
    st.image(output, width="stretch")


aircraft_count = len(detected_boxes)

st.markdown(
    f"""
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Aircraft detected</div>
        <div class="metric-value">{aircraft_count}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Orientations resolved</div>
        <div class="metric-value">{orientation_count}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Inference time</div>
        <div class="metric-value">{elapsed:.2f}s</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)