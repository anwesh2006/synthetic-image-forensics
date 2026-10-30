# synthetic-image-forensics

A lightweight Streamlit demo for synthetic image detection that combines:
- Hugging Face model artifact download
- ONNX runtime inference
- Simple, built-in XAI overlay for visual interpretation

## Repository layout

```text
synthetic-image-forensics/
├── .streamlit/
│   └── config.toml          # Streamlit UI theme and port settings
├── streamlit_app.py         # Core application: HF weight loader, ONNX inference, and XAI
├── requirements.txt         # Container dependencies for Streamlit Community Cloud
├── README.md                # Model documentation, architecture details, and benchmarks
└── .gitignore               # Git exclusions
```

## Architecture

1. **Weight loading**: `hf_hub_download` fetches ONNX model weights from a Hugging Face repository.
2. **Preprocessing**: Uploaded RGB images are resized and normalized into NCHW tensors.
3. **Inference**: `onnxruntime` executes the model on CPU.
4. **Explainability (XAI)**: A gradient-based visual overlay highlights regions that contribute to high-confidence predictions.

## Benchmarks

The app is intentionally model-agnostic. Throughput and latency depend on:
- selected ONNX model size,
- image resolution,
- runtime environment.

Use Streamlit Community Cloud app logs and local profiling to capture reproducible benchmark numbers for your deployed model.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
