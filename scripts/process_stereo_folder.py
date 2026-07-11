#!/usr/bin/env python3
"""Process folders matched by identical relative image filename."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from carvision.recorded_stereo import matched_image_pairs, save_diagnostics
from recorded_cli_common import add_adapter_arguments, adapter_from_args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-every", type=int, default=1,
                        help="save full diagnostics every N frames; zero saves summaries only")
    add_adapter_arguments(parser)
    args = parser.parse_args()
    pairs = matched_image_pairs(args.left_dir, args.right_dir)
    adapter = adapter_from_args(args)
    summaries = []
    wall_start = perf_counter()
    for index, (left_path, right_path) in enumerate(pairs):
        left, right, result = adapter.process_files(left_path, right_path)
        frame_output = args.output_dir / f"{index:06d}_{left_path.stem}"
        if args.save_every > 0 and index % args.save_every == 0:
            save_diagnostics(frame_output, left, right, result)
        summaries.append({"index": index, "left": str(left_path), "right": str(right_path),
                          "valid_fraction": float(result.valid_mask.mean()),
                          "point_count": len(result.points_vehicle_m),
                          "timings_ms": result.timings_ms})
    elapsed = perf_counter() - wall_start
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequence_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    totals = [entry["timings_ms"]["total"] for entry in summaries]
    print(f"frames={len(pairs)} wall_fps={len(pairs)/elapsed:.2f} "
          f"median_pipeline_ms={np.median(totals):.1f}")


if __name__ == "__main__":
    main()

