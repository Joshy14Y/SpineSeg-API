import numpy as np
from scipy.interpolate import UnivariateSpline

from inference.types import CobbEndpoint, CobbResult


class CobbAngleEstimator:
    """Estimates the Cobb angle from vertebra centers using spline fitting."""

    _MIN_CENTERS = 3
    _SLOPE_EPSILON = 1e-6
    _CURVATURE_SAMPLES = 300

    def __init__(
        self,
        k: int = 2,
        s: int = 5,
        min_separation: float = 40.0,
        apex_tolerance: float = 30.0,
    ):
        """Configure spline degree, smoothing factor, and minimum endpoint
        separation."""
        self.k = k
        self.s = s
        self.min_separation = min_separation
        self.apex_tolerance = apex_tolerance

    def __call__(self, centers: np.ndarray) -> tuple[UnivariateSpline, CobbResult]:
        """Return the fitted spline and Cobb angle result, or raise if undetermined."""
        if len(centers) < self._MIN_CENTERS:
            raise ValueError("Not enough centers to fit a spline.")
        spline = self._fit_spline(centers)
        cobb = self._compute_cobb(spline, centers)
        return spline, cobb

    def _fit_spline(self, centers: np.ndarray) -> UnivariateSpline:
        """Fit a spline through vertebra centers mapping y → x."""
        ys, xs = centers[:, 1], centers[:, 0]
        return UnivariateSpline(ys, xs, k=self.k, s=self.s)

    def _compute_cobb(
        self, spline: UnivariateSpline, centers: np.ndarray
    ) -> CobbResult:
        """Compute the Cobb angle using perpendicular slopes for clinical accuracy."""
        upper, lower = self._find_endpoints(spline, centers)
        s1 = self._perpendicular_slope(upper.slope)
        s2 = self._perpendicular_slope(lower.slope)
        angle = self._angle_between_slopes(s1, s2)
        return CobbResult(
            CobbEndpoint(upper.x, upper.y, s1),
            CobbEndpoint(lower.x, lower.y, s2),
            angle,
        )

    def _perpendicular_slope(self, slope: float) -> float:
        """Return the slope perpendicular to the given slope."""
        if abs(slope) < self._SLOPE_EPSILON:
            return 1e6
        return -1 / slope

    def _angle_between_slopes(self, s1: float, s2: float) -> float:
        """Compute the angle in degrees between two slopes."""
        denom = 1 + s1 * s2
        if abs(denom) < self._SLOPE_EPSILON:
            return 90.0
        tangent = abs((s2 - s1) / denom)
        radians = np.arctan(tangent)
        degrees = np.degrees(radians)
        return float(degrees)

    def _find_endpoints(
        self, spline: UnivariateSpline, centers: np.ndarray
    ) -> tuple[CobbEndpoint, CobbEndpoint]:
        """Find the upper and lower end vertebrae with steepest slopes."""
        centers = centers[np.argsort(centers[:, 1])]
        ys, xs = centers[:, 1], centers[:, 0]
        slopes = spline.derivative(1)(ys)
        abs_slopes = np.abs(slopes)

        upper_mask, lower_mask = self._split_by_apex(spline, ys)
        if not upper_mask.any() or not lower_mask.any():
            raise ValueError("Could not split vertebrae by apex.")

        upper_idx = self._steepest_idx(upper_mask, abs_slopes)
        lower_idx = self._steepest_idx(lower_mask, abs_slopes)

        upper = CobbEndpoint(xs[upper_idx], ys[upper_idx], slopes[upper_idx])
        lower = CobbEndpoint(xs[lower_idx], ys[lower_idx], slopes[lower_idx])

        if abs(upper.y - lower.y) < self.min_separation:
            raise ValueError("Endpoints too close.")

        return upper, lower

    def _split_by_apex(
        self, spline: UnivariateSpline, ys: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split vertebrae into upper and lower groups relative to the apex."""
        y_apex = self._find_apex(spline, ys.min(), ys.max())
        upper_mask = ys < y_apex
        lower_mask = ys > y_apex
        if not upper_mask.any():
            upper_mask = ys < (y_apex + self.apex_tolerance)
        if not lower_mask.any():
            lower_mask = ys > (y_apex - self.apex_tolerance)
        return upper_mask, lower_mask

    def _find_apex(self, spline: UnivariateSpline, y_min: float, y_max: float) -> float:
        """Find the y coordinate of maximum spline curvature."""
        ys = np.linspace(y_min, y_max, self._CURVATURE_SAMPLES)
        curvature = spline.derivative(2)(ys)
        abs_curvature = np.abs(curvature)
        max_curvature_idx = np.argmax(abs_curvature)
        return ys[max_curvature_idx]

    def _steepest_idx(self, mask: np.ndarray, abs_slopes: np.ndarray) -> int:
        """Return the index of the steepest vertebra within a masked half."""
        idx = np.argmax(abs_slopes[mask])
        return np.where(mask)[0][idx]
