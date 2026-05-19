from pathlib import Path

import onnxruntime as ort


def load_model(model_path: Path) -> ort.InferenceSession:
    """Load an ONNX model and return an inference session."""
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
