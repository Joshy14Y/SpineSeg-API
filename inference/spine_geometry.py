import numpy as np
from scipy.signal import find_peaks


class SpineGeometry:
    """Detects vertebra centers by finding valleys in the spine's horizontal profile."""

    def __init__(
        self,
        distance: int = 8,
        prominence: int = 20,
        max_valleys: int = 16,
        min_pixels: int = 50,
    ):
        """Configure peak detection and region filtering parameters."""
        self.distance = distance
        self.prominence = prominence
        self.max_valleys = max_valleys
        self.min_pixels = min_pixels

    def __call__(self, mask: np.ndarray) -> np.ndarray:
        """Return vertebra centers as (x, y) pairs sorted top to bottom."""
        seg_mask = mask > 0
        valleys = self._find_valleys(seg_mask)
        bands = self._build_bands(valleys, seg_mask.shape[0])
        return self._extract_centers(seg_mask, bands)

    def _find_valleys(self, mask: np.ndarray) -> np.ndarray:
        """Find row indices where vertebrae are separated, sorted by prominence."""
        profile = mask.sum(axis=1)
        valleys, props = find_peaks(
            -profile, distance=self.distance, prominence=self.prominence
        )
        if len(valleys) < 1:
            raise ValueError("No valleys found.")
        order = np.argsort(props["prominences"])[::-1]
        return np.sort(valleys[order[: self.max_valleys]])

    def _build_bands(self, valleys: np.ndarray, img_height: int) -> list[tuple]:
        """Slice the image into vertical bands between consecutive valleys."""
        cuts = [0] + valleys.tolist() + [img_height]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def _extract_centers(self, mask: np.ndarray, bands: list[tuple]) -> np.ndarray:
        """Compute the centroid of each band, skipping sparse regions."""
        raw = [self._band_center(mask, y0, y1) for y0, y1 in bands]
        centers = np.array([c for c in raw if c is not None])
        if len(centers) < 1:
            raise ValueError("No vertebra centers detected.")
        sorted_idxs = np.argsort(centers[:, 1])
        return centers[sorted_idxs]

    def _band_center(self, mask: np.ndarray, y0: int, y1: int) -> tuple | None:
        """Return the centroid of a band, or None if too sparse."""
        region = mask[y0:y1]
        if region.sum() < self.min_pixels:
            return None
        ys, xs = np.where(region)
        cx, cy = xs.mean(), y0 + ys.mean()
        if not self._is_valid_center(mask, cx, cy):
            return None
        return (cx, cy)

    def _is_valid_center(self, mask: np.ndarray, cx: float, cy: float) -> bool:
        """Check that the centroid lands on a foreground pixel."""
        row = min(int(round(cy)), mask.shape[0] - 1)
        col = min(int(round(cx)), mask.shape[1] - 1)
        return bool(mask[row, col])
