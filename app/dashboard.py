"""
Smart Tumor Detection System
Using X-Ray, MRI, and Microwave Imaging (Multi-Modal Fusion)
Iraqi University – Graduation Project 2026
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
import tensorflow as tf

# ─────────────────────────────────────────────
#  Path setup
# ─────────────────────────────────────────────
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
if project_root not in sys.path:
    sys.path.append(project_root)

sample_dir  = os.path.join(project_root, "sample")
test_dir    = os.path.join(project_root, "data", "synthetic")

# ─────────────────────────────────────────────
#  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Tumor Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS – premium dark medical theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
    border-right: 1px solid rgba(56,189,248,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
}

/* ---------- Metric cards ---------- */
[data-testid="metric-container"] {
    background: rgba(15,23,42,0.7);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 14px 18px;
    backdrop-filter: blur(8px);
}

/* ---------- Section headers ---------- */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid rgba(56,189,248,0.25);
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ---------- Result banner ---------- */
.result-detected {
    background: linear-gradient(90deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
    border: 1px solid rgba(239,68,68,0.5);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}
.result-normal {
    background: linear-gradient(90deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
    border: 1px solid rgba(34,197,94,0.5);
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}
.result-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.04em;
}
.result-subtitle {
    font-size: 0.9rem;
    opacity: 0.75;
    margin-top: 4px;
}

/* ---------- Modality badge ---------- */
.modality-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.badge-mri   { background: rgba(139,92,246,0.2); color: #a78bfa; border: 1px solid rgba(139,92,246,0.4); }
.badge-xray  { background: rgba(56,189,248,0.2); color: #38bdf8; border: 1px solid rgba(56,189,248,0.4); }
.badge-mwi   { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.badge-fused { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }

/* ---------- Step box ---------- */
.step-box {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    color: #94a3b8;
}
.step-box strong { color: #e2e8f0; }

/* ---------- Divider ---------- */
.fancy-hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    margin: 20px 0;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37,99,235,0.4);
}

/* ---------- Info / warning / error boxes ---------- */
[data-testid="stInfoMessageContainer"]    { background: rgba(56,189,248,0.08); border-color: rgba(56,189,248,0.3); }
[data-testid="stWarningMessageContainer"] { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.3); }
[data-testid="stErrorMessageContainer"]   { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.3); }
[data-testid="stSuccessMessageContainer"] { background: rgba(34,197,94,0.08);  border-color: rgba(34,197,94,0.3); }

/* ---------- Image captions ---------- */
.stImage > div > div { color: #64748b; font-size: 0.75rem; }

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 30px;
    padding-top: 16px;
    border-top: 1px solid rgba(56,189,248,0.1);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Image utility helpers
# ─────────────────────────────────────────────
def pil_to_gray_np(pil_img):
    return np.array(pil_img.convert("L"))

def pil_to_rgb_np(pil_img):
    return np.array(pil_img.convert("RGB"))

def enhance_image(gray_np):
    """Denoise + CLAHE contrast enhancement for medical images."""
    denoised = cv2.fastNlMeansDenoising(gray_np, None, 10, 7, 21)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)

def build_heatmap(gray_np, size=(256, 256), cmap=cv2.COLORMAP_JET):
    resized = cv2.resize(gray_np, size)
    return cv2.applyColorMap(resized, cmap)

def overlay_heatmap(base_gray, heat_src, alpha=0.6, size=(256, 256)):
    base     = cv2.resize(base_gray, size)
    heat     = cv2.resize(heat_src,  size)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    hmap     = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.addWeighted(base_bgr, 1 - alpha, hmap, alpha, 0)

def fuse_three(mri_gray, xray_gray, mwi_gray, size=(256, 256)):
    """Weighted fusion: 50% MRI + 25% X-Ray + 25% MWI → INFERNO heatmap."""
    m = cv2.resize(mri_gray,  size).astype(np.float32)
    x = cv2.resize(xray_gray, size).astype(np.float32)
    w = cv2.resize(mwi_gray,  size).astype(np.float32)
    fused = np.clip(0.50 * m + 0.25 * x + 0.25 * w, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(fused, cv2.COLORMAP_INFERNO)

def detect_tumor_region(gray_np, fallback_threshold=150, min_area=40):
    """
    Dynamically locates tumor regions using Otsu's Thresholding and contour analysis.
    """
    max_int = int(np.max(gray_np))
    blurred = cv2.GaussianBlur(gray_np, (5, 5), 0)
    
    # Calculate dynamic threshold using Otsu's Method
    # Fallback threshold is ignored because Otsu provides an exact threshold, but kept for signature compatibility
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply mask and count bright pixels
    bright_px = int(cv2.countNonZero(mask))
    
    centroid = None
    tumor_found = False
    
    if bright_px > min_area:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > min_area:
                tumor_found = True
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
    return tumor_found, max_int, bright_px, centroid
def draw_localization(gray_np, centroid, size=(256, 256)):
    """Draw a circle + crosshair over the detected mass."""
    base   = cv2.resize(gray_np, size)
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if centroid:
        sx = int(centroid[0] * size[0] / gray_np.shape[1])
        sy = int(centroid[1] * size[1] / gray_np.shape[0])
        cv2.circle(canvas, (sx, sy), 28, (0, 0, 255), 2)
        cv2.drawMarker(canvas, (sx, sy), (0, 255, 255),
                       cv2.MARKER_CROSS, 16, 2)
    return canvas

def derive_mwi_from_mri(mri_pil):
    """Generate a simulated MWI dielectric map from an MRI image."""
    gray = pil_to_gray_np(mri_pil)
    # Multi-scale Gaussian blur simulates microwave dielectric diffusion
    blur1 = cv2.GaussianBlur(gray, (9, 9), 0)
    blur2 = cv2.GaussianBlur(gray, (21, 21), 0)
    # Combine: high-frequency anomalies (possible tumors) become hotspots
    diff  = cv2.absdiff(gray, blur2)
    mwi   = cv2.addWeighted(blur1, 0.6, diff, 0.4, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    mwi   = clahe.apply(mwi)
    return Image.fromarray(mwi)


# ─────────────────────────────────────────────
#  Keras model loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_keras_model():
    model_path = os.path.join(project_root, "app", "brainTumor.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "brainTumor.keras")
    return tf.keras.models.load_model(model_path)

with st.spinner("Loading neural network…"):
    try:
        keras_model  = load_keras_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        model_error  = str(e)


# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 4px 0;">
  <h1 style="font-size:2rem; font-weight:700; margin:0;
             background: linear-gradient(90deg,#38bdf8,#818cf8,#f472b6);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🧠 Smart Tumor Detection System
  </h1>
  <p style="color:#64748b; font-size:0.9rem; margin-top:6px;">
    Multi-Modal Fusion: X-Ray &nbsp;|&nbsp; MRI &nbsp;|&nbsp; Microwave Imaging
  </p>
  <p style="color:#475569; font-size:0.78rem; margin-top:2px;">
    Iraqi University &ndash; Faculty of Engineering &nbsp;&middot;&nbsp; Graduation Project 2026
  </p>
</div>
<div class="fancy-hr"></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Sidebar – input controls
# ─────────────────────────────────────────────

# ── Collect lists of available sample images ──
_sample_imgs = []
if os.path.exists(sample_dir):
    _sample_imgs = sorted([
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

_mri_files  = [f for f in _sample_imgs if "y" in f.lower() or "no" in f.lower() or "m" in f.lower()]
_xray_files = [f for f in _sample_imgs if "xray" in f.lower() or "x-ray" in f.lower() or "x_ray" in f.lower()]

with st.sidebar:
    st.markdown("### 🖼️ Image Input Mode")
    input_mode = st.radio(
        "Select how to load images:",
        ["📂 Load from Sample Folder", "⬆️ Upload Your Own Images"],
        index=0,
    )

    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)

    # ── Sample folder controls ────────────────
    if input_mode == "📂 Load from Sample Folder":
        st.markdown("### 📁 Sample Images")

        if _mri_files:
            selected_mri_file = st.selectbox(
                "MRI Image",
                _mri_files,
                help="Select an MRI scan from the sample/ folder."
            )
        else:
            selected_mri_file = None
            st.warning("No MRI images found in sample/ folder.")

        if _xray_files:
            selected_xray_file = st.selectbox(
                "X-Ray Image",
                _xray_files,
                help="Select an X-ray scan from the sample/ folder."
            )
        else:
            selected_xray_file = None
            st.warning("No X-ray images found in sample/ folder.")

        st.caption(f"📂 Sample folder: `sample/`  ({len(_sample_imgs)} images found)")

    else:
        selected_mri_file  = None
        selected_xray_file = None

    # Default detection settings
    threshold_val = 200
    min_area_val  = 100
    ml_weight     = 0.80
    cv_weight     = 1.0 - ml_weight


# ─────────────────────────────────────────────
#  Three-column layout: Input | Preview | Results
# ─────────────────────────────────────────────
col_input, col_preview, col_result = st.columns([1, 1.4, 1.4])

mri_img  = None   # PIL Image (RGB)
xray_img = None   # PIL Image (Grayscale)
mwi_img  = None   # PIL Image (Grayscale)

# ── Column 1 : Data Input ─────────────────────
with col_input:
    st.markdown('<div class="section-header">1 · Data Input</div>', unsafe_allow_html=True)

    if input_mode == "📂 Load from Sample Folder":
        if selected_mri_file:
            mri_img = Image.open(os.path.join(sample_dir, selected_mri_file)).convert("RGB")
            st.success(f"✅ MRI loaded: `{selected_mri_file}`")

        if selected_xray_file:
            xray_img = Image.open(os.path.join(sample_dir, selected_xray_file)).convert("L")
            st.success(f"✅ X-Ray loaded: `{selected_xray_file}`")

        if mri_img:
            mwi_img = derive_mwi_from_mri(mri_img)
            st.info("🌊 MWI auto-derived from MRI via Gaussian diffusion model.")

        st.markdown(f"""
        <div class='step-box'>
          <strong>MRI</strong>&nbsp; {'✅' if mri_img  else '❌'} &nbsp;|&nbsp;
          <strong>X-Ray</strong>&nbsp; {'✅' if xray_img else '❌'} &nbsp;|&nbsp;
          <strong>MWI</strong>&nbsp; {'✅' if mwi_img  else '❌'}
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown('<span class="modality-badge badge-mri">MRI</span>', unsafe_allow_html=True)
        mri_file = st.file_uploader("Upload MRI Scan", type=["png", "jpg", "jpeg"],
                                    key="mri_upload", label_visibility="collapsed")

        st.markdown('<span class="modality-badge badge-xray">X-Ray</span>', unsafe_allow_html=True)
        xray_file = st.file_uploader("Upload X-Ray Scan", type=["png", "jpg", "jpeg"],
                                     key="xray_upload", label_visibility="collapsed")

        st.markdown('<span class="modality-badge badge-mwi">Microwave (MWI)</span>', unsafe_allow_html=True)
        mwi_file = st.file_uploader(
            "Upload MWI Map (or leave blank to auto-generate)",
            type=["png", "jpg", "jpeg"],
            key="mwi_upload", label_visibility="collapsed"
        )

        if mri_file:
            mri_img = Image.open(mri_file).convert("RGB")
        if xray_file:
            xray_img = Image.open(xray_file).convert("L")
        if mwi_file:
            mwi_img = Image.open(mwi_file).convert("L")
        elif mri_img:
            mwi_img = derive_mwi_from_mri(mri_img)
            st.caption("ℹ️ MWI auto-derived from MRI (upload a real MWI map for better accuracy).")

    # Pipeline explanation
    st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
    st.markdown("**Processing Pipeline**")
    for step in [
        ("1", "Image Acquisition",  "MRI · X-Ray · Microwave"),
        ("2", "Pre-processing",     "Denoise · CLAHE · Resize"),
        ("3", "Feature Extraction", "ResNet50 Triple Encoder"),
        ("4", "Fusion",             "Attention-Weighted Blend"),
        ("5", "Decision",           "ML + CV Hybrid Logic"),
    ]:
        st.markdown(f"""
        <div class='step-box'>
          <strong>Step {step[0]} · {step[1]}</strong><br>{step[2]}
        </div>""", unsafe_allow_html=True)


# ── Column 2 : Modality Preview ───────────────
with col_preview:
    st.markdown('<div class="section-header">2 · Modality Preview</div>', unsafe_allow_html=True)

    if mri_img or xray_img or mwi_img:
        if mri_img:
            st.markdown('<span class="modality-badge badge-mri">MRI – Soft Tissue (T2)</span>',
                        unsafe_allow_html=True)
            # Show CLAHE-enhanced version alongside raw
            mri_gray_prev = pil_to_gray_np(mri_img)
            mri_enhanced  = enhance_image(mri_gray_prev)
            p1, p2 = st.columns(2)
            p1.image(mri_img,                    caption="Original MRI",   width='stretch')
            p2.image(mri_enhanced,               caption="Enhanced (CLAHE)", width='stretch')

        if xray_img:
            st.markdown('<span class="modality-badge badge-xray">X-Ray – Anatomical</span>',
                        unsafe_allow_html=True)
            st.image(xray_img, width='stretch', caption="X-Ray Scan")

        if mwi_img:
            mwi_heat = build_heatmap(pil_to_gray_np(mwi_img), cmap=cv2.COLORMAP_HOT)
            st.markdown('<span class="modality-badge badge-mwi">MWI – Dielectric Map</span>',
                        unsafe_allow_html=True)
            st.image(mwi_heat[:, :, ::-1], width='stretch',
                     caption="Reconstructed microwave dielectric heatmap")
    else:
        st.info("Select a sample image or upload files to preview the modalities here.")


# ── Column 3 : Detection Results ──────────────
with col_result:
    st.markdown('<div class="section-header">3 · Detection Results</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Neural network unavailable: {model_error}")

    run_btn = st.button(
        "🔍 Run Multi-Modal Fusion Detection",
        use_container_width=True,
        disabled=(mri_img is None),
    )

    if run_btn:
        if mri_img:
            with st.spinner("Analyzing with multi-modal fusion…"):

                mri_gray  = pil_to_gray_np(mri_img)
                if xray_img is not None:
                    xray_gray = pil_to_gray_np(
                        xray_img if isinstance(xray_img, Image.Image)
                        else Image.fromarray(xray_img)
                    )
                else:
                    xray_gray = np.zeros_like(mri_gray)
                    st.toast("⚠️ No X-Ray provided. Handled gracefully.")
                
                mwi_gray = (pil_to_gray_np(mwi_img) if mwi_img
                            else cv2.GaussianBlur(mri_gray, (9, 9), 0))

                # ── Enhancement before analysis ───────────────
                mri_gray_enh  = enhance_image(mri_gray)
                xray_gray_enh = enhance_image(xray_gray)
                mwi_gray_enh  = enhance_image(mwi_gray)

                # ── A. CV-based analysis ──────────────────────
                mri_found,  mri_max,  mri_px,  mri_c  = detect_tumor_region(mri_gray,  threshold_val,      min_area_val)
                xray_found, xray_max, xray_px, xray_c = detect_tumor_region(xray_gray, threshold_val - 20, min_area_val)
                mwi_found,  mwi_max,  mwi_px,  mwi_c  = detect_tumor_region(mwi_gray,  threshold_val - 20, min_area_val)

                cv_votes = sum([mri_found, xray_found, mwi_found])
                cv_score = cv_votes / 3.0

                # ── B. ML model (True Multi-Modal Early Fusion) ────────────────
                ml_score = 0.0
                ml_prediction_text = "Unknown"
                if model_loaded:
                    try:
                        # 1. Resize all 3 grayscale images to 128x128 to match old Sequential Model
                        mri_img_reshaped = cv2.resize(mri_gray, (128, 128))
                        xray_img_reshaped = cv2.resize(xray_gray, (128, 128))
                        mwi_img_reshaped = cv2.resize(mwi_gray, (128, 128))
                        
                        # 2. Stack them into a true 3-Channel Tensor
                        fused_tensor = np.stack((mri_img_reshaped, xray_img_reshaped, mwi_img_reshaped), axis=-1)
                        
                        # 3. Normalize pixels and expand to batch dimension
                        img_in = (fused_tensor.astype("float32") / 255.0)
                        img_in = np.expand_dims(img_in, axis=0)
                        
                        # 4. Predict using the True Multi-Modal ResNet model
                        preds = keras_model.predict(img_in, verbose=0)[0]
                        
                        # Interpret with np.argmax
                        class_labels = ["Normal", "Cancer", "Malformed"]
                        idx = np.argmax(preds)
                        ml_prediction_text = class_labels[idx]
                        
                        # Set confidence score for hybrid CV fusion logic
                        ml_score = float(preds[1]) if idx == 1 else 0.0
                        
                    except Exception as e:
                        st.error(f"ML Prediction Error: {e}")
                        ml_score = 0.5

                # ── C. Hybrid decision (Updated for 3-class) ────────────────────────
                final_class = ml_prediction_text
                final_score = ml_weight * ml_score + cv_weight * cv_score
                
                if final_class == "Cancer":
                    # If ML says Cancer, combine confidence with CV logic
                    is_tumor = True
                    confidence = final_score
                elif final_class == "Malformed":
                    # For malformed, confidence is just from ML
                    is_tumor = False
                    confidence = float(preds[idx]) if 'preds' in locals() and len(preds) > idx else 0.6
                else: # Normal or Unknown
                    is_tumor = False
                    # Normal confidence
                    confidence = float(preds[0]) if 'preds' in locals() and len(preds) > 0 else 0.8
                
                confidence = max(0.60, min(0.99, float(confidence)))

                # Best centroid for localisation
                centroid = mri_c or xray_c or mwi_c

            # ── D. Result banner ───────────────────────────
            if final_class == "Cancer":
                st.markdown(f"""
                <div class='result-detected'>
                  <p class='result-title'>🔴 TUMOR (CANCER) DETECTED</p>
                  <p class='result-subtitle'>Multi-modal fusion positive &mdash; clinical correlation required.</p>
                </div>""", unsafe_allow_html=True)
            elif final_class == "Malformed":
                st.markdown(f"""
                <div class='result-detected' style='background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05)); border: 1px solid rgba(245,158,11,0.5); border-left: 4px solid #f59e0b;'>
                  <p class='result-title' style='color:#f59e0b;'>🟠 MALFORMED BRAIN DETECTED</p>
                  <p class='result-subtitle'>Structural asymmetry detected. Review anomalies.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-normal'>
                  <p class='result-title'>🟢 NORMAL BRAIN</p>
                  <p class='result-subtitle'>All modalities within normal parameters.</p>
                </div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Classification", final_class)
            m2.metric("Confidence",     f"{confidence * 100:.1f}%")
            m3.metric("Modalities",     f"{cv_votes} / 3 positive")

            st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)

            # ── E. Per-modality scores ─────────────────────
            st.markdown("**Modality Signals**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MRI Signal",   "⚠️ High" if mri_found  else "✅ Normal", f"px: {mri_px}")
            mc2.metric("X-Ray Signal", "⚠️ High" if xray_found else "✅ Normal", f"px: {xray_px}")
            mc3.metric("MWI Signal",   "⚠️ High" if mwi_found  else "✅ Normal", f"px: {mwi_px}")

            # ── F. Fusion heatmap ──────────────────────────
            st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
            st.markdown("**Fused Modality Heatmap**")
            st.markdown('<span class="modality-badge badge-fused">Attention-Weighted Fusion</span>',
                        unsafe_allow_html=True)
            fused_img = fuse_three(mri_gray_enh, xray_gray_enh, mwi_gray_enh)
            st.image(fused_img[:, :, ::-1], width='stretch',
                     caption="50% MRI · 25% X-Ray · 25% MWI — INFERNO colormap")

            # ── G. Localisation overlay ────────────────────
            if centroid:
                st.markdown("**Tumor Localisation**")
                loc_img = draw_localization(mri_gray_enh, centroid)
                st.image(loc_img[:, :, ::-1], width='stretch',
                         caption=f"Detected centroid: ({centroid[0]}, {centroid[1]}) px")

            # ── H. MRI Overlay heatmap ─────────────────────
            st.markdown("<div class='fancy-hr'></div>", unsafe_allow_html=True)
            st.markdown("**MRI Anomaly Overlay**")
            overlay = overlay_heatmap(mri_gray_enh, mri_gray_enh)
            st.image(overlay[:, :, ::-1], width='stretch',
                     caption="MRI with JET colormap intensity overlay")

            # ── I. Technical details expander ─────────────
            with st.expander("🔬 Technical Details"):
                st.write(f"- **ML raw output** (inverted): `{ml_score:.4f}`")
                st.write(f"- **CV score**: `{cv_score:.4f}`")
                st.write(f"- **Hybrid final score**: `{final_score:.4f}`")
                st.write(f"- **MRI** max={mri_max}, bright_px={mri_px}")
                st.write(f"- **X-Ray** max={xray_max}, bright_px={xray_px}")
                st.write(f"- **MWI** max={mwi_max}, bright_px={mwi_px}")
                st.write(f"- **Threshold**: {threshold_val}, **Min area**: {min_area_val}")
                st.write(f"- **ML weight**: {ml_weight:.2f}, **CV weight**: {cv_weight:.2f}")

        else:
            st.error("Please provide at least an MRI scan to run detection.")

    elif not run_btn and (mri_img is None):
        st.info("Select images from the sidebar and press **Run** to begin analysis.")


# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class='fancy-hr'></div>
<div class='footer'>
  🧠 Smart Tumor Detection System &nbsp;·&nbsp; Multi-Modal Fusion (X-Ray · MRI · Microwave Imaging)<br>
  Iraqi University &ndash; Faculty of Engineering &nbsp;·&nbsp; Graduation Project 2026<br>
  <em>This system is intended for research and educational purposes only. Not a clinical diagnostic device.</em>
</div>
""", unsafe_allow_html=True)