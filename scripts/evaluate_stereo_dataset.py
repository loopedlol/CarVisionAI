#!/usr/bin/env python3
"""Evaluate every annotated frame and sequence in a dataset manifest."""

import argparse
import json
from pathlib import Path

from carvision.evaluation import evaluate_dataset
from evaluation_cli_common import load_dataset_and_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sgbm-config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-frame-diagnostics", action="store_true")
    args = parser.parse_args()
    dataset, config = load_dataset_and_config(args.dataset, args.sgbm_config)
    report = evaluate_dataset(dataset, config, args.output_dir,
                              save_frame_diagnostics=not args.no_frame_diagnostics)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

