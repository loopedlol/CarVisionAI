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


@dataclass(frozen=True)
class NoisyMultisensorScene:
    """Noisy measurements of shared front and side obstacles."""

    disparity_px: NDArray[np.float64]
    calibration: StereoCalibration
    tof_ranges_m: NDArray[np.float64]
    tof_azimuth_rad: NDArray[np.float64]


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


def make_noisy_multisensor_scene(
    height: int = 240, width: int = 320, *, seed: int = 7
) -> NoisyMultisensorScene:
    """Create overlapping stereo/ToF data with noise, holes, and gross outliers."""
    if height < 48 or width < 64:
        raise ValueError("noisy scene dimensions must be at least 48x64")
    rng = np.random.default_rng(seed)
    scale = width / 320.0
    calibration = StereoCalibration(280.0 * scale, 280.0 * scale,
                                    (width - 1) / 2, (height - 1) / 2, 0.12)
    depth = np.full((height, width), 7.0)
    depth[height // 3: 5 * height // 6, 2 * width // 5: 3 * width // 5] = 3.0
    depth[height // 2: 4 * height // 5, 3 * width // 4: 9 * width // 10] = 4.5
    disparity = calibration.fx_px * calibration.baseline_m / depth
    disparity += rng.normal(0.0, 0.12, depth.shape)
    disparity[rng.random(depth.shape) < 0.06] = np.nan
    disparity[rng.random(depth.shape) < 0.015] = 0.0
    disparity[rng.random(depth.shape) < 0.003] = rng.choice([-2.0, 0.02, 80.0])

    azimuth = np.linspace(-np.deg2rad(55), np.deg2rad(55), 361)
    ranges = 7.0 / np.cos(azimuth)
    center = np.abs(azimuth) < np.deg2rad(11)
    right = (azimuth > np.deg2rad(22)) & (azimuth < np.deg2rad(34))
    ranges[center] = 3.0 / np.cos(azimuth[center])
    ranges[right] = 4.5 / np.cos(azimuth[right])
    ranges += rng.normal(0.0, 0.025, ranges.shape)
    ranges[rng.random(ranges.shape) < 0.05] = np.nan
    ranges[rng.choice(len(ranges), 5, replace=False)] = np.array([0.0, -1.0, np.inf, 0.08, 30.0])
    return NoisyMultisensorScene(disparity, calibration, ranges, azimuth)
