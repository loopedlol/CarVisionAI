#!/usr/bin/env python3
"""Benchmark a recorded pair after warm-up, reporting each pipeline stage."""

import argparse
from statistics import median

import cv2

from recorded_cli_common import add_adapter_arguments, adapter_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--iterations", type=int, default=10)
    add_adapter_arguments(parser)
    args = parser.parse_args()
    adapter = adapter_from_args(args)
    left = cv2.imread(args.left, cv2.IMREAD_UNCHANGED)
    right = cv2.imread(args.right, cv2.IMREAD_UNCHANGED)
    if left is None or right is None or args.iterations < 1:
        raise ValueError("valid images and positive iterations are required")
    adapter.process_arrays(left, right)  # warm caches and OpenCV paths
    timings: dict[str, list[float]] = {}
    for _ in range(args.iterations):
        result = adapter.process_arrays(left, right)
        for stage, elapsed in result.timings_ms.items():
            timings.setdefault(stage, []).append(elapsed)
    print(" ".join(f"{stage}_median={median(values):.1f}ms" for stage, values in timings.items()))


if __name__ == "__main__":
    main()

