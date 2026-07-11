"""Deterministic synthetic measurements for pipeline development."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .stereo import StereoCalibration


@dataclass(frozen=True)
class MockStereoScene:
    disparity_px: NDArray[np.float64]
    expected_depth_m: NDArray[np.float64]
    calibration: StereoCalibration


def make_mock_stereo_scene(height: int = 48, width: int = 64) -> MockStereoScene:
    """Create a front wall with a nearer rectangular obstacle and missing pixels."""
    if height < 8 or width < 8:
        raise ValueError("mock image dimensions must each be at least 8")
    calibration = StereoCalibration(80.0, 80.0, (width - 1) / 2, (height - 1) / 2, 0.12)
    depth = np.full((height, width), 6.0, dtype=np.float64)
    depth[height // 3: 3 * height // 4, width // 3: 2 * width // 3] = 2.5
    disparity = calibration.fx_px * calibration.baseline_m / depth
    disparity[:2, :] = np.nan
    disparity[:, :2] = 0.0
    expected = depth.copy()
    expected[~(np.isfinite(disparity) & (disparity > 0))] = np.nan
    return MockStereoScene(disparity, expected, calibration)


def make_mock_tof_scan(ray_count: int = 181) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a planar scan with a wall at y=5 m and a central obstacle at y=2 m."""
    if ray_count < 3:
        raise ValueError("ray_count must be at least 3")
    azimuth = np.linspace(-np.pi / 3, np.pi / 3, ray_count)
    ranges = 5.0 / np.cos(azimuth)
    obstacle = np.abs(azimuth) < np.deg2rad(10.0)
    ranges[obstacle] = 2.0 / np.cos(azimuth[obstacle])
    ranges[::37] = np.nan
    return ranges, azimuth

