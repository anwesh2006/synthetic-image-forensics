import io
from typing import Sequence

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


st.set_page_config(page_title="Synthetic Image Forensics", layout="wide")


@st.cache_resource(show_spinner=False)
def load_onnx_session(repo_id: str, filename: str):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def preprocess_image(image: Image.Image, image_size: int = 224) -> np.ndarray:
    resized = image.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


def run_inference(session, image_tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image_tensor})[0]
    return np.squeeze(output)


def generate_xai_overlay(image: Image.Image, confidence: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1, :]))
    gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :, :]))
    grad = (gx + gy).mean(axis=2)
    grad = grad / (grad.max() + 1e-8)

    heat = np.zeros_like(arr)
    heat[:, :, 0] = grad * 255.0 * float(confidence)

    blended = 0.65 * arr + 0.35 * heat
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def read_labels(raw_text: str) -> Sequence[str]:
    labels = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return labels if labels else ["synthetic", "real"]


def main() -> None:
    st.title("Synthetic Image Forensics")
    st.caption("Upload an image to run ONNX inference with model weights loaded from Hugging Face.")

    with st.sidebar:
        st.header("Model Settings")
        repo_id = st.text_input("Hugging Face Repo ID", value="owner/model-repo")
        model_file = st.text_input("ONNX Model Filename", value="model.onnx")
        labels_text = st.text_area("Class Labels (one per line)", value="synthetic\nreal")

    uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])
    if not uploaded:
        st.info("Upload an image to begin analysis.")
        return

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", use_column_width=True)

    labels = read_labels(labels_text)

    try:
        session = load_onnx_session(repo_id=repo_id, filename=model_file)
        logits = run_inference(session, preprocess_image(image))
        probs = softmax(np.asarray(logits, dtype=np.float32))
    except Exception as exc:  # pragma: no cover
        st.error(f"Inference failed: {exc}")
        return

    predicted_index = int(np.argmax(probs))
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"class_{predicted_index}"
    confidence = float(probs[predicted_index])

    with col2:
        st.metric("Prediction", predicted_label)
        st.metric("Confidence", f"{confidence:.2%}")
        st.image(generate_xai_overlay(image, confidence), caption="XAI Overlay", use_column_width=True)

    st.subheader("Class Probabilities")
    for idx, score in enumerate(probs.tolist()):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        st.write(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
