#!/usr/bin/env python3
"""Microbenchmark encoder, IMU, and external pose estimator updates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from carvision.control import Pose2D, WheelSpeeds
from carvision.pose_estimation import (EncoderMeasurement, EstimatorMode,
                                      ExternalPoseMeasurement, ImuYawRateMeasurement,
                                      PoseEstimator)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("outputs/pose_fusion/benchmark.json"))
    args = parser.parse_args(); estimator = PoseEstimator(EstimatorMode.ENCODER_IMU_EXTERNAL)
    measurements = {
        "encoder": lambda i: estimator.process_encoder(EncoderMeasurement(i * .01, WheelSpeeds(5, 5))),
        "imu": lambda i: estimator.process_imu(ImuYawRateMeasurement(args.iterations * .01 + i * .01, 0.01)),
        "external": lambda i: estimator.process_external_pose(ExternalPoseMeasurement(
            2 * args.iterations * .01 + i * .01, estimator.pose, np.eye(3) * .01)),
    }
    report = {}
    for name, operation in measurements.items():
        samples = []
        for i in range(args.iterations):
            begin = perf_counter(); operation(i); samples.append((perf_counter() - begin) * 1000)
        report[name] = {"median_ms": float(np.median(samples)),
                        "p99_ms": float(np.percentile(samples, 99))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
