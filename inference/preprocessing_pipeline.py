import albumentations as A
import cv2
import numpy as np

from inference.types import PreprocessingResult


class PreprocessingPipeline:
    """Converts raw image bytes into a normalized grayscale tensor ready for
    inference."""

    _TARGET_HEIGHT = 1024
    _TARGET_WIDTH = 512
    _CLAHE_CLIP_LIMIT = 2

    def __init__(self, mean: float = 0.3974, std: float = 0.0802):
        """Build resize and normalization pipelines using training-time statistics."""
        self.resize_pipeline = A.Compose(
            [
                A.LongestMaxSize(max_size=self._TARGET_HEIGHT),
                A.PadIfNeeded(
                    min_height=self._TARGET_HEIGHT, min_width=self._TARGET_WIDTH, fill=0
                ),
                A.CenterCrop(height=self._TARGET_HEIGHT, width=self._TARGET_WIDTH),
            ]
        )
        self.normalize_pipeline = A.Compose(
            [
                A.CLAHE(clip_limit=self._CLAHE_CLIP_LIMIT, p=1),
                A.Normalize(mean=[mean], std=[std]),
            ]
        )

    def __call__(self, img_bytes: bytes) -> PreprocessingResult:
        """Decode, resize, normalize, and return inference-ready tensor with display
        image and crop box."""
        img = self._decode(img_bytes)
        original_w = int(img.shape[1] * (self._TARGET_HEIGHT / img.shape[0]))
        np_img, tensor = self._preprocess(img)
        crop_box = self._crop_box(np_img.shape[1], original_w)
        return PreprocessingResult(tensor, np_img, crop_box)

    def _decode(self, img_bytes: bytes) -> np.ndarray:
        """Decode raw bytes into a grayscale numpy array."""
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes.")
        return img

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resize, pad, crop, and normalize. Returns display image and tensor."""
        np_img = self.resize_pipeline(image=img)["image"]
        tensor = self.normalize_pipeline(image=np_img)["image"]
        return np_img, tensor[np.newaxis, np.newaxis, ...]

    def _crop_box(self, padded_w: int, original_w: int) -> tuple[int, int, int, int]:
        """Compute the bounding box that strips padding from the resized image."""
        x_start = (padded_w - original_w) // 2
        return (x_start, 0, x_start + original_w, self._TARGET_HEIGHT)
