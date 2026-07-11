#!/usr/bin/env python3
"""Run a synthetic rotating-ToF scan through local mapping."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from carvision.mapping import OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_mock_tof_scan
from carvision.tof import ranges_to_vehicle_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tof"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranges, azimuth = make_mock_tof_scan()
    points = ranges_to_vehicle_points(ranges, azimuth, max_range_m=10.0)
    grid = OccupancyGrid(OccupancyGridConfig(y_min_m=-1.0, y_max_m=7.0))
    grid.update(points)
    np.save(args.output_dir / "ranges_m.npy", ranges)
    np.save(args.output_dir / "points_vehicle_m.npy", points)
    np.save(args.output_dir / "occupancy.npy", grid.data)
    cv2.imwrite(str(args.output_dir / "occupancy.png"), grid.preview_u8())
    print(f"valid returns: {len(points)}; occupied cells: {np.count_nonzero(grid.data == 100)}")


if __name__ == "__main__":
    main()

