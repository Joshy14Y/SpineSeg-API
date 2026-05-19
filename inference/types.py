from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class PreprocessingResult:
    tensor: np.ndarray
    np_img: np.ndarray
    crop_box: tuple[int, int, int, int]


@dataclass
class SegmentationResult:
    mask: np.ndarray
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
class Vertebra:
    uid: str
    class_id: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "class_id": self.class_id,
            "confidence": self.confidence,
        }


@dataclass
class PipelineResult:
    mask_img: Image.Image
    annotated_img: Image.Image
    vertebrae: list[Vertebra]
    cobb_angle: float
