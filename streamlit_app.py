import streamlit as st
import numpy as np
import cv2
import torch
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Synthetic Image Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Constants & Model Caching
# ---------------------------------------------------------
HF_REPO_ID = "anweshh/synthetic-image-forensics"
ONNX_FILENAME = "forensic_model.onnx"

@st.cache_resource(show_spinner="Fetching forensic weights from Hugging Face...")
def load_onnx_session():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ONNX_FILENAME)
    session = ort.InferenceSession(
        model_path, 
        providers=['CPUExecutionProvider']
    )
    return session

session = load_onnx_session()

# ---------------------------------------------------------
# Forensic Signal Extraction
# ---------------------------------------------------------
def preprocess_dual_stream(img_rgb):
    """
    Constructs normalized spatial tensor and its centered 
    2D FFT log-magnitude spectrum for dual-stream inference.
    """
    resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)
    norm = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    spatial_tensor = np.transpose(norm, (2, 0, 1)).astype(np.float32)[np.newaxis, ...]

    with torch.no_grad():
        x_torch = torch.from_numpy(spatial_tensor)
        fft = torch.fft.fft2(x_torch, norm="ortho")
        fft_shift = torch.fft.fftshift(fft)
        spectral_tensor = torch.log(torch.abs(fft_shift) + 1e-7).numpy()

    return spatial_tensor, spectral_tensor, resized

def compute_azimuthal_psd(img_rgb):
    """
    Radially integrates the 2D power spectrum into a 1D 
    azimuthal profile to expose synthetic high-frequency anomalies.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    psd2D = np.abs(fshift) ** 2

    y, x = np.indices(psd2D.shape)
    center = np.array([(y.max() - y.min()) / 2.0, (x.max() - x.min()) / 2.0])
    r = np.hypot(x - center[1], y - center[0]).astype(int)

    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    max_radius = min(center).astype(int)
    
    return np.log10(radial_profile[:max_radius] + 1e-8)

def compute_2d_spectrum_display(img_rgb):
    """
    Computes visual 2D FFT magnitude spectrum for UI display.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude_spectrum

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("Diagnostic Settings")
    temperature = st.slider(
        "Temperature Scaling (T)",
        min_value=1.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Calibrates output probabilities against distribution drift."
    )
    decision_threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Probability boundary to flag an image as synthetic."
    )
    st.divider()
    st.markdown(f"**Repository:** `{HF_REPO_ID}`")
    st.markdown(f"**Runtime:** ONNX CPU Engine")

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------
st.title("Synthetic Image Forensics & Frequency Verification")
st.caption("Dual-Stream ConvNeXt-Tiny Spatial & 2D FFT Spectral Diagnostic System")

uploaded_file = st.file_uploader(
    "Upload an image for forensic examination (PNG, JPG, JPEG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 1. Pipeline Execution
    spatial_input, spectral_input, resized_rgb = preprocess_dual_stream(img_rgb)

    raw_logits = session.run(None, {
        'spatial_input': spatial_input,
        'spectral_input': spectral_input
    })[0][0]

    # Temperature-calibrated sigmoid
    calibrated_prob = float(1.0 / (1.0 + np.exp(-raw_logits / temperature)))
    is_synthetic = calibrated_prob >= decision_threshold

    # 2. Results Header
    st.divider()
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        if is_synthetic:
            st.error("Verdict: SYNTHETIC / MANIPULATED")
        else:
            st.success("Verdict: AUTHENTIC PHOTOGRAPH")
            
    with metric_col2:
        st.metric(
            label="Synthetic Probability",
            value=f"{calibrated_prob * 100:.2f}%"
        )
        
    with metric_col3:
        st.metric(
            label="Raw Logit Score",
            value=f"{raw_logits:.4f}"
        )

    # 3. Spatial & Frequency Panels
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Spatial Domain Inspection")
        st.image(img_rgb, caption="Analyzed Input Frame", use_container_width=True)

    with col_right:
        st.subheader("Frequency Domain Profiling")
        tab1, tab2 = st.tabs(["1D Radial PSD Profile", "2D FFT Magnitude"])

        with tab1:
            radial_psd = compute_azimuthal_psd(resized_rgb)
            fig_psd, ax_psd = plt.subplots(figsize=(6, 4))
            line_color = "#FF4B4B" if is_synthetic else "#00C853"
            
            ax_psd.plot(radial_psd, color=line_color, lw=2, label="Sample Profile")
            ax_psd.set_xlabel("Spatial Frequency Radius (DC -> Nyquist)")
            ax_psd.set_ylabel("Log Power Spectral Density (PSD)")
            ax_psd.grid(True, linestyle="--", alpha=0.3)
            ax_psd.legend(loc="upper right")
            st.pyplot(fig_psd)
            plt.close(fig_psd)

        with tab2:
            spec_2d = compute_2d_spectrum_display(resized_rgb)
            fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
            im = ax_spec.imshow(spec_2d, cmap="inferno")
            ax_spec.axis("off")
            fig_spec.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
            st.pyplot(fig_spec)
            plt.close(fig_spec)
