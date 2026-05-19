import numpy as np
import onnxruntime as ort
from PIL import Image

from inference.cobb_angle_estimator import CobbAngleEstimator
from inference.confidence_estimator import ConfidenceEstimator
from inference.preprocessing_pipeline import PreprocessingPipeline
from inference.segmentation_model import SegmentationModel
from inference.spine_geometry import SpineGeometry
from inference.spine_renderer import SpineRenderer
from inference.types import PipelineResult


class InferencePipeline:
    """Runs the full spine analysis pipeline on a single image."""

    def __init__(self, model: ort.InferenceSession):
        """Initialize all pipeline components."""
        self.preprocess = PreprocessingPipeline()
        self.segment = SegmentationModel(model)
        self.geometry = SpineGeometry()
        self.cobb = CobbAngleEstimator()
        self.confidence = ConfidenceEstimator()
        self.renderer = SpineRenderer()

    def __call__(self, img_bytes: bytes) -> PipelineResult:
        """Run inference and return results with rendered image."""
        preprocessed = self.preprocess(img_bytes)
        segmentation = self.segment(preprocessed.tensor)
        mask = segmentation.mask
        centers = self.geometry(mask)
        spline, cobb = self.cobb(centers)
        vertebrae = self.confidence(mask, segmentation.probs, centers)
        annotated_img = self.renderer(preprocessed.np_img, mask, centers, spline, cobb)
        mask_img = Image.fromarray(mask.astype(np.uint8))
        return PipelineResult(
            mask_img=mask_img.crop(preprocessed.crop_box),
            annotated_img=annotated_img.crop(preprocessed.crop_box),
            vertebrae=vertebrae,
            cobb_angle=cobb.angle,
        )
