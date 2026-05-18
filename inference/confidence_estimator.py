import numpy as np
from scipy.ndimage import label
from inference.types import VertebraConfidence


class ConfidenceEstimator:
    """Computes per-instance confidence scores for each detected vertebra."""

    def __call__(
        self, id_mask: np.ndarray, probs: np.ndarray, centers: np.ndarray
    ) -> list[VertebraConfidence]:
        """Score all detected vertebrae and return sorted valid results."""
        if len(centers) < 1:
            raise ValueError("Not a single center provided.")
        scored = [self._process(c, id_mask, probs) for c in centers]
        valid = [r for r in scored if r.confidence > 0]
        if len(valid) < 1:
            raise ValueError("No vertebrae with valid confidence.")
        return sorted(valid, key=lambda v: v.vertebra_id)

    def _process(
        self, center: tuple, mask: np.ndarray, probs: np.ndarray
    ) -> VertebraConfidence:
        """Build a confidence entry for a single vertebra center."""
        row, col = self._to_pixel(center, mask.shape)
        vertebra_id = int(mask[row, col])
        component = self._get_component(mask, vertebra_id, row, col)
        confidence = self._compute_confidence(probs, vertebra_id, component)
        return VertebraConfidence(vertebra_id, confidence)

    def _to_pixel(self, center: tuple, shape: tuple) -> tuple[int, int]:
        """Convert floating-point center coordinates to clamped pixel indices."""
        x, y = center
        h, w = shape
        row = max(0, min(h - 1, int(round(y))))
        col = max(0, min(w - 1, int(round(x))))
        return row, col

    def _get_component(
        self, mask: np.ndarray, vertebra_id: int, row: int, col: int
    ) -> np.ndarray:
        """Return the connected blob containing the center pixel, or empty if background."""
        labeled, _ = label(mask == vertebra_id)
        instance_id = labeled[row, col]
        if instance_id < 1:
            return np.array([])
        return labeled == instance_id

    def _compute_confidence(
        self, probs: np.ndarray, vertebra_id: int, component: np.ndarray
    ) -> float:
        """Average the vertebra's predicted probabilities over its component pixels."""
        vertebra_probs = probs[vertebra_id]
        if len(component) < 1:
            return 0.0
        return float(vertebra_probs[component].mean())
