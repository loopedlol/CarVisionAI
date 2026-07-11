#!/usr/bin/env python3
"""Run stereo, ToF, or both through the common occupancy mapper."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from carvision.frames import camera_to_vehicle
from carvision.mapping import OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_mock_stereo_scene, make_mock_tof_scan
from carvision.stereo import backproject_depth, disparity_to_depth, valid_points
from carvision.tof import ranges_to_vehicle_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", choices=("stereo", "tof", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/combined"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clouds: list[np.ndarray] = []
    if args.sensor in ("stereo", "both"):
        scene = make_mock_stereo_scene()
        depth = disparity_to_depth(scene.disparity_px, scene.calibration)
        clouds.append(valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration))))
    if args.sensor in ("tof", "both"):
        ranges, azimuth = make_mock_tof_scan()
        clouds.append(ranges_to_vehicle_points(ranges, azimuth, max_range_m=10.0))
    points = np.concatenate(clouds)
    grid = OccupancyGrid(OccupancyGridConfig(y_min_m=-1.0, y_max_m=8.0))
    grid.update(points)
    np.save(args.output_dir / "points_vehicle_m.npy", points)
    np.save(args.output_dir / "occupancy.npy", grid.data)
    cv2.imwrite(str(args.output_dir / "occupancy.png"), grid.preview_u8())
    print(f"sensor={args.sensor}; points={len(points)}; occupied={np.count_nonzero(grid.data == 100)}; free={np.count_nonzero(grid.data == 0)}")


if __name__ == "__main__":
    main()

