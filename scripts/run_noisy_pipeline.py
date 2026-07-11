#!/usr/bin/env python3
"""Run the overlapping noisy stereo and ToF mock scene."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from carvision.frames import camera_to_vehicle
from carvision.mapping import OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_noisy_multisensor_scene
from carvision.stereo import backproject_depth, disparity_to_depth, valid_points
from carvision.tof import ranges_to_vehicle_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/noisy"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = make_noisy_multisensor_scene()
    depth = disparity_to_depth(scene.disparity_px, scene.calibration, max_depth_m=12.0)
    stereo = valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration)))
    tof = ranges_to_vehicle_points(scene.tof_ranges_m, scene.tof_azimuth_rad, max_range_m=12.0)
    grid = OccupancyGrid(OccupancyGridConfig(-6, 6, -1, 9, 0.1, -0.4, 1.4))
    grid.update(stereo)
    grid.update(tof)
    np.save(args.output_dir / "depth_m.npy", depth)
    np.save(args.output_dir / "stereo_points_vehicle_m.npy", stereo)
    np.save(args.output_dir / "tof_points_vehicle_m.npy", tof)
    np.save(args.output_dir / "occupancy.npy", grid.data)
    cv2.imwrite(str(args.output_dir / "occupancy.png"), grid.preview_u8())
    print(f"stereo={len(stereo)} tof={len(tof)} occupied={np.count_nonzero(grid.data == 100)} free={np.count_nonzero(grid.data == 0)}")


if __name__ == "__main__":
    main()

