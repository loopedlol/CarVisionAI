#!/usr/bin/env python3
"""Save only original and rectified stereo pairs with epipolar guides."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from carvision.recorded_stereo import StereoRigCalibration, RecordedStereoAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = RecordedStereoAdapter(StereoRigCalibration.load(args.calibration))
    left = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError("could not read input pair")
    rect_left, rect_right = adapter.rectify(left, right)
    original = np.hstack((left, right))
    rectified = cv2.cvtColor(np.hstack((rect_left, rect_right)), cv2.COLOR_GRAY2BGR)
    for y in range(20, rectified.shape[0], 40):
        cv2.line(rectified, (0, y), (rectified.shape[1] - 1, y), (0, 255, 0), 1)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "original_pair.png"), original)
    cv2.imwrite(str(args.output / "rectified_pair_guides.png"), rectified)
    print(f"saved rectification diagnostics to {args.output}")


if __name__ == "__main__":
    main()

