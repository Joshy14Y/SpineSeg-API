import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
from inference.types import CobbResult

from constants.id2rgb import ID2RGB


class SpineRenderer:
    """Renders segmentation overlay, vertebra centers, spline, and Cobb angle lines."""

    _SPLINE_SAMPLES = 300
    _TANGENT_SAMPLES = 100
    _CENTER_RADIUS = 5
    _ENDPOINT_RADIUS = 6
    _LINE_THICKNESS = 3

    def __init__(
        self,
        alpha: float = 0.3,
        center_color: tuple = (255, 255, 0),
        spline_color: tuple = (0, 255, 255),
        cobb_color: tuple = (255, 0, 0),
        endpoint_color: tuple = (255, 255, 255),
    ):
        """Configure overlay transparency and annotation colors."""
        self.alpha = alpha
        self.center_color = center_color
        self.spline_color = spline_color
        self.cobb_color = cobb_color
        self.endpoint_color = endpoint_color

    def __call__(
        self,
        image: np.ndarray,
        preds: np.ndarray,
        centers: np.ndarray,
        spline: UnivariateSpline,
        cobb: CobbResult,
    ) -> Image.Image:
        """Return a PIL Image with all spine annotations rendered."""
        canvas = self._prepare_canvas(image)
        self._draw_segmentation(canvas, preds)
        self._draw_centers(canvas, centers)
        self._draw_spline(canvas, spline, centers)
        self._draw_cobb(canvas, cobb)
        return Image.fromarray(canvas)

    def _prepare_canvas(self, image: np.ndarray) -> np.ndarray:
        """Normalize and convert grayscale image to RGB canvas."""
        img = image.copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def _draw_segmentation(self, canvas: np.ndarray, preds: np.ndarray) -> None:
        """Overlay segmentation mask using the colormap."""
        colored = np.zeros_like(canvas)
        for class_id in np.unique(preds):
            mask_class = preds == class_id
            colored[mask_class] = ID2RGB.get(class_id, (255, 255, 255))
        mask = preds > 0
        canvas[mask] = (
            (1 - self.alpha) * canvas[mask] + self.alpha * colored[mask]
        ).astype(np.uint8)

    def _draw_centers(self, canvas: np.ndarray, centers: np.ndarray) -> None:
        """Draw vertebra centers."""
        for x, y in centers:
            cv2.circle(
                canvas,
                (int(x), int(y)),
                radius=self._CENTER_RADIUS,
                color=self.center_color,
                thickness=-1,
            )

    def _draw_spline(
        self, canvas: np.ndarray, spline: UnivariateSpline, centers: np.ndarray
    ) -> None:
        """Draw the fitted spline."""
        ys = np.linspace(centers[:, 1].min(), centers[:, 1].max(), self._SPLINE_SAMPLES)
        xs = spline(ys)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(
                canvas,
                tuple(pts[i]),
                tuple(pts[i + 1]),
                color=self.spline_color,
                thickness=self._LINE_THICKNESS,
            )

    def _draw_cobb(self, canvas: np.ndarray, cobb: CobbResult) -> None:
        """Draw Cobb angle tangent lines and endpoint markers."""
        x1, y1, s1 = cobb.upper.x, cobb.upper.y, cobb.upper.slope
        x2, y2, s2 = cobb.lower.x, cobb.lower.y, cobb.lower.slope
        line1 = self._tangent_line(x1, y1, s1)
        line2 = self._tangent_line(x2, y2, s2)
        self._draw_line(canvas, line1)
        self._draw_line(canvas, line2)
        cv2.circle(
            canvas,
            (int(x1), int(y1)),
            radius=self._ENDPOINT_RADIUS,
            color=self.endpoint_color,
            thickness=-1,
        )
        cv2.circle(
            canvas,
            (int(x2), int(y2)),
            radius=self._ENDPOINT_RADIUS,
            color=self.endpoint_color,
            thickness=-1,
        )

    def _tangent_line(
        self, x0: float, y0: float, slope: float, length: int = 200
    ) -> np.ndarray:
        """Compute pixel coordinates of a tangent line through a point."""
        dy = np.linspace(-length, length, self._TANGENT_SAMPLES)
        dx = slope * dy
        return np.stack([x0 + dx, y0 + dy], axis=1).astype(np.int32)

    def _draw_line(self, canvas: np.ndarray, pts: np.ndarray) -> None:
        """Draw a polyline on the canvas."""
        for i in range(len(pts) - 1):
            cv2.line(
                canvas,
                tuple(pts[i]),
                tuple(pts[i + 1]),
                color=self.cobb_color,
                thickness=self._LINE_THICKNESS,
            )
