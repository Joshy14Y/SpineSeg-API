import numpy as np
from PIL import Image
import torch
from inference.cobb_angle_estimator import CobbAngleEstimator
from inference.confidence_estimator import ConfidenceEstimator
from inference.preprocessing_pipeline import PreprocessingPipeline
from inference.segmentation_model import SegmentationModel
from inference.spine_geometry import SpineGeometry
from inference.spine_renderer import SpineRenderer
from inference.types import PipelineResult


class InferencePipeline:
    """Runs the full spine analysis pipeline on a single image."""

    def __init__(self, model: torch.nn.Module):
        """Initialize all pipeline components."""
        self.preprocess = PreprocessingPipeline()
        self.segment = SegmentationModel(model)
        self.geometry = SpineGeometry()
        self.cobb = CobbAngleEstimator()
        self.confidence = ConfidenceEstimator()
        self.renderer = SpineRenderer()

    def __call__(self, image_bytes: bytes) -> PipelineResult:
        """Run inference and return results with rendered image."""
        preprocessed = self.preprocess(image_bytes)
        segmentation = self.segment(preprocessed.tensor)
        id_mask = segmentation.id_mask
        centers = self.geometry(id_mask)
        spline, cobb = self.cobb(centers)
        vertebrae = self.confidence(id_mask, segmentation.probs, centers)
        render = self.renderer(preprocessed.image_np, id_mask, centers, spline, cobb)
        id_mask_img = Image.fromarray(id_mask.astype(np.uint8))
        return PipelineResult(
            mask_img=id_mask_img.crop(preprocessed.crop_box),
            annotated_img=render.crop(preprocessed.crop_box),
            vertebrae=vertebrae,
            cobb_angle=cobb.angle,
        )
