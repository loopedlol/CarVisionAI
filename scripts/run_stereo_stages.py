#!/usr/bin/env python3
"""Run and save each synthetic stereo geometry stage."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from carvision.frames import camera_to_vehicle
from carvision.mock import make_mock_stereo_scene
from carvision.stereo import backproject_depth, disparity_to_depth, valid_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stereo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = make_mock_stereo_scene()
    depth = disparity_to_depth(scene.disparity_px, scene.calibration)
    camera_cloud = backproject_depth(depth, scene.calibration)
    vehicle_points = valid_points(camera_to_vehicle(camera_cloud))
    np.save(args.output_dir / "disparity_px.npy", scene.disparity_px)
    np.save(args.output_dir / "depth_m.npy", depth)
    np.save(args.output_dir / "points_vehicle_m.npy", vehicle_points)
    cv2.imwrite(str(args.output_dir / "depth_preview.png"), _depth_preview(depth))
    print(f"valid points: {len(vehicle_points)}; depth range: {np.nanmin(depth):.2f}-{np.nanmax(depth):.2f} m")


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    image = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        image[valid] = np.clip(255 * (1 - depth[valid] / np.nanmax(depth)), 0, 255)
    return image


if __name__ == "__main__":
    main()

