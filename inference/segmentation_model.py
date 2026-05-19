import numpy as np
import onnxruntime as ort

from inference.types import SegmentationResult


class SegmentationModel:
    """Runs inference and produces a class probability map and a vertebra ID mask."""

    _NUM_VERTEBRAE = 17
    _FG_THRESHOLD = 0.5
    _LOG_EPSILON = 1e-6

    def __init__(self, model: ort.InferenceSession, sigma: float = 0.03):
        """Wrap a segmentation model for inference."""
        self.model = model
        self.sigma = sigma
        self.vertebra_centers = np.linspace(0.05, 0.95, self._NUM_VERTEBRAE)

    def __call__(self, tensor: np.ndarray) -> SegmentationResult:
        """Return probs (C, H, W) and id_mask (H, W) for the given image array."""
        probs = self._forward(tensor)
        fg_mask = probs[1:].sum(axis=0) > self._FG_THRESHOLD
        id_mask = (np.argmax(probs[1:], axis=0) + 1) * fg_mask
        return SegmentationResult(id_mask, probs)

    def _forward(self, tensor: np.ndarray) -> np.ndarray:
        """Run a forward pass and return per-class softmax probabilities."""
        model_input = {"input": tensor.astype(np.float32)}
        z, fg_logits, refine_logits = self.model.run(None, model_input)
        probs = self._build_probs(z, fg_logits)
        final_logits = refine_logits + np.log(probs + self._LOG_EPSILON)
        return self._softmax(final_logits, axis=1)[0]

    def _build_probs(self, z: np.ndarray, fg_logits: np.ndarray) -> np.ndarray:
        """Combine foreground probability and soft vertebra assignment into a prior."""
        fg = self._sigmoid(fg_logits)
        vertebra_probs = self._soft_assign(z) * fg
        bg = 1 - fg
        return np.concatenate([bg, vertebra_probs], axis=1)

    def _soft_assign(self, z: np.ndarray) -> np.ndarray:
        """Assign each pixel to a vertebra center using Gaussian similarity."""
        centers = self.vertebra_centers.reshape(1, -1, 1, 1)
        z_expanded = z - centers
        logits = -(z_expanded**2) / self.sigma
        return self._softmax(logits, axis=1)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        """Numerically stable softmax along the given axis."""
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Element-wise sigmoid activation."""
        return 1 / (1 + np.exp(-x))
