from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class PreprocessingResult:
    tensor: torch.Tensor
    image_np: np.ndarray
    crop_box: tuple[int, int, int, int]


@dataclass
class SegmentationResult:
    id_mask: np.ndarray
    probs: np.ndarray


@dataclass
class CobbEndpoint:
    x: float
    y: float
    slope: float


@dataclass
class CobbResult:
    upper: CobbEndpoint
    lower: CobbEndpoint
    angle: float


@dataclass
class VertebraConfidence:
    vertebra_id: int
    confidence: float

    def to_dict(self) -> dict:
        return {"id": self.vertebra_id, "confidence": self.confidence}


@dataclass
class PipelineResult:
    mask_img: Image.Image
    annotated_img: Image.Image
    vertebrae: list[VertebraConfidence]
    cobb_angle: float
