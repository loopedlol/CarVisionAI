#!/usr/bin/env python3
"""Process one recorded left/right image pair and save diagnostics."""

import argparse
from pathlib import Path

from carvision.recorded_stereo import save_diagnostics
from recorded_cli_common import add_adapter_arguments, adapter_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_adapter_arguments(parser)
    args = parser.parse_args()
    adapter = adapter_from_args(args)
    left, right, result = adapter.process_files(args.left, args.right)
    save_diagnostics(args.output_dir, left, right, result)
    print(f"points={len(result.points_vehicle_m)} valid={result.valid_mask.mean():.1%} "
          + " ".join(f"{key}={value:.1f}ms" for key, value in result.timings_ms.items()))


if __name__ == "__main__":
    main()

