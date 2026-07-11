#!/usr/bin/env python3
"""Compare a bounded explicit set of SGBM/filter configurations."""

import argparse
import json
from pathlib import Path

from carvision.evaluation import EvaluationDataset, compare_configurations
from evaluation_cli_common import load_named_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--configs", required=True,
                        help="JSON mapping up to 12 names to partial SGBM configurations")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset = EvaluationDataset.load(args.dataset)
    report = compare_configurations(dataset, load_named_configs(args.configs, dataset), args.output_dir)
    print(json.dumps(report["configurations"], indent=2))


if __name__ == "__main__":
    main()

