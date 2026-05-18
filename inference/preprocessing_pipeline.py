import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from inference.types import PreprocessingResult


class PreprocessingPipeline:
    """Converts raw image bytes into a normalized grayscale tensor ready for inference."""

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
                ToTensorV2(),
            ]
        )

    def __call__(self, image_bytes: bytes) -> PreprocessingResult:
        """Decode image bytes and return preprocessed tensor, display image, and crop box."""
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Failed to decode image bytes.")

        scale = self._TARGET_HEIGHT / image.shape[0]
        original_w = int(image.shape[1] * scale)

        image_np = self.resize_pipeline(image=image)["image"]
        tensor = self.normalize_pipeline(image=image_np)["image"]

        padded_w = image_np.shape[1]
        x_start = (padded_w - original_w) // 2
        crop_box = (x_start, 0, x_start + original_w, self._TARGET_HEIGHT)

        return PreprocessingResult(tensor, image_np, crop_box)
