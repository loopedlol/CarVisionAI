#!/usr/bin/env python3
"""Calibrate a stereo rig from matched checkerboard image folders."""

import argparse
from pathlib import Path

import numpy as np

from carvision.frames import CAMERA_OPTICAL_TO_VEHICLE, RigidTransform
from carvision.stereo_calibration import CheckerboardConfig, calibrate_stereo_folders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--columns", type=int, required=True, help="checkerboard inner corners across")
    parser.add_argument("--rows", type=int, required=True, help="checkerboard inner corners down")
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--camera-translation-m", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument("--camera-rpy-deg", type=float, nargs=3, default=(0, 0, 0),
                        help="roll pitch yaw applied in vehicle axes after canonical optical conversion")
    parser.add_argument("--min-pairs", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roll, pitch, yaw = np.deg2rad(args.camera_rpy_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    extrinsic = RigidTransform(rz @ ry @ rx @ CAMERA_OPTICAL_TO_VEHICLE.rotation,
                               np.asarray(args.camera_translation_m))
    report = calibrate_stereo_folders(
        args.left_dir, args.right_dir,
        CheckerboardConfig(args.columns, args.rows, args.square_size_m),
        camera_to_vehicle=extrinsic, min_pairs=args.min_pairs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.calibration.save(args.output)
    print(f"used={report.used_pairs} rejected={report.rejected_pairs} "
          f"left_rms={report.left_rms_px:.3f}px right_rms={report.right_rms_px:.3f}px "
          f"stereo_rms={report.stereo_rms_px:.3f}px baseline="
          f"{np.linalg.norm(report.calibration.right_from_left_translation_m):.4f}m")


if __name__ == "__main__":
    main()
