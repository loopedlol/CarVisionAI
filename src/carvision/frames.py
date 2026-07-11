"""Coordinate-frame conversions."""

import numpy as np
from numpy.typing import NDArray


def camera_to_vehicle(points_camera: NDArray[np.floating]) -> NDArray[np.float64]:
    """Convert (..., 3) camera-frame points to vehicle frame.

    Camera is x-right, y-down, z-forward. Vehicle is x-right, y-forward,
    z-up. NaN values are preserved.
    """
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points_camera must have shape (..., 3)")
    return np.stack((points[..., 0], points[..., 2], -points[..., 1]), axis=-1)

