#!/usr/bin/env python3
"""Benchmark vector geometry and grid integration for a dense mock frame."""

from time import perf_counter

from carvision.frames import camera_to_vehicle
from carvision.mapping import OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_noisy_multisensor_scene
from carvision.stereo import backproject_depth, disparity_to_depth, valid_points


def main() -> None:
    scene = make_noisy_multisensor_scene(480, 640)
    start = perf_counter()
    depth = disparity_to_depth(scene.disparity_px, scene.calibration, max_depth_m=12.0)
    points = valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration)))
    geometry_s = perf_counter() - start
    grid = OccupancyGrid(OccupancyGridConfig(-6, 6, -1, 9, 0.1))
    start = perf_counter()
    grid.update(points)
    mapping_s = perf_counter() - start
    print(f"pixels={depth.size} valid_points={len(points)} geometry_ms={geometry_s*1000:.1f} mapping_ms={mapping_s*1000:.1f} total_ms={(geometry_s+mapping_s)*1000:.1f}")


if __name__ == "__main__":
    main()
