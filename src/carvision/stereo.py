"""Rectified stereo geometry."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StereoCalibration:
    """Minimal calibration for a rectified pinhole stereo pair."""

    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    baseline_m: float

    def __post_init__(self) -> None:
        values = (self.fx_px, self.fy_px, self.baseline_m)
        if not all(np.isfinite(values)) or any(value <= 0 for value in values):
            raise ValueError("focal lengths and baseline must be finite and positive")
        if not np.isfinite((self.cx_px, self.cy_px)).all():
            raise ValueError("principal point must be finite")


def disparity_to_depth(
    disparity_px: NDArray[np.floating], calibration: StereoCalibration
) -> NDArray[np.float64]:
    """Convert rectified disparity to optical-axis depth in metres.

    Non-finite and non-positive disparities produce NaN depth.
    """
    disparity = np.asarray(disparity_px, dtype=np.float64)
    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(disparity) & (disparity > 0.0)
    depth[valid] = calibration.fx_px * calibration.baseline_m / disparity[valid]
    return depth


def backproject_depth(
    depth_m: NDArray[np.floating], calibration: StereoCalibration
) -> NDArray[np.float64]:
    """Back-project a depth image to an HxWx3 camera-frame organized cloud."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array")
    v, u = np.indices(depth.shape, dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 0.0)
    safe_depth = np.where(valid, depth, np.nan)
    x = (u - calibration.cx_px) * safe_depth / calibration.fx_px
    y = (v - calibration.cy_px) * safe_depth / calibration.fy_px
    return np.stack((x, y, safe_depth), axis=-1)


def valid_points(organized_points: NDArray[np.floating]) -> NDArray[np.float64]:
    """Flatten an organized cloud and remove non-finite points."""
    points = np.asarray(organized_points, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("organized_points must have shape (H, W, 3)")
    flat = points.reshape(-1, 3)
    return flat[np.isfinite(flat).all(axis=1)]

