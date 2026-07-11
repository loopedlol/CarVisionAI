"""Shared compact CLI options for recorded stereo scripts."""

import argparse

from carvision.mapping import OccupancyGridConfig
from carvision.recorded_stereo import RecordedStereoAdapter, SGBMConfig, StereoRigCalibration


def add_adapter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--min-disparity", type=int, default=0)
    parser.add_argument("--num-disparities", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--speckle-window-size", type=int, default=100)
    parser.add_argument("--lr-max-diff", type=float, default=1.5,
                        help="left-right consistency threshold in pixels; negative disables")
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--map-width", type=float, default=10.0)
    parser.add_argument("--map-forward", type=float, default=20.0)
    parser.add_argument("--map-resolution", type=float, default=0.1)
    parser.add_argument("--min-obstacle-height", type=float, default=-0.4)
    parser.add_argument("--max-obstacle-height", type=float, default=1.5)


def adapter_from_args(args: argparse.Namespace) -> RecordedStereoAdapter:
    sgbm = SGBMConfig(
        min_disparity=args.min_disparity, num_disparities=args.num_disparities,
        block_size=args.block_size, uniqueness_ratio=args.uniqueness_ratio,
        speckle_window_size=args.speckle_window_size,
        left_right_max_diff_px=None if args.lr_max_diff < 0 else args.lr_max_diff,
        min_depth_m=args.min_depth, max_depth_m=args.max_depth,
    )
    grid = OccupancyGridConfig(
        -args.map_width / 2, args.map_width / 2, -1.0, args.map_forward - 1.0,
        args.map_resolution, args.min_obstacle_height, args.max_obstacle_height,
    )
    return RecordedStereoAdapter(StereoRigCalibration.load(args.calibration), sgbm, grid)

