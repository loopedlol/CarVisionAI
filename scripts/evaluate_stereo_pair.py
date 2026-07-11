#!/usr/bin/env python3
"""Evaluate one annotated frame selected from a dataset manifest."""

import argparse
import json
from pathlib import Path

from carvision.recorded_stereo import RecordedStereoAdapter, StereoRigCalibration
from carvision.evaluation import evaluate_frame
from evaluation_cli_common import load_dataset_and_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--sgbm-config")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset, config = load_dataset_and_config(args.dataset, args.sgbm_config)
    scene, frame = dataset.find(args.scene, args.frame)
    adapter = RecordedStereoAdapter(StereoRigCalibration.load(dataset.calibration_path), config)
    metrics, _ = evaluate_frame(adapter, scene, frame, args.output_dir)
    print(json.dumps({"scene": scene.scene_id, "frame": frame.frame_id,
                      "valid_fraction": metrics["valid_disparity_fraction"],
                      "rectification_p95_px": metrics["vertical_error_p95_px"],
                      "regions": metrics["regions"], "timings_ms": metrics["timings_ms"]}, indent=2))


if __name__ == "__main__":
    main()

