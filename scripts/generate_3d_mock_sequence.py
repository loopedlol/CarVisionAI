#!/usr/bin/env python3
"""Generate and headlessly inspect the end-to-end mock 3D snapshot sequence."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from carvision.visualization3d import (ViewerConfig, ViewerControls, prepare_snapshot,
                                      render_prepared_preview, save_snapshot)
from carvision.visualization_mock import make_mock_visualization_sequence


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--frames",type=int,default=40);parser.add_argument("--output-dir",type=Path,default=Path("outputs/viewer3d/mock_sequence"));args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    sequence=make_mock_visualization_sequence(args.frames);config=ViewerConfig(max_points=100000)
    previews=[];indices=sorted({0,len(sequence)//2,len(sequence)-1});summary=[]
    snapshot_dir=args.output_dir/"snapshots";snapshot_dir.mkdir(exist_ok=True)
    for item in sequence:
        save_snapshot(item,snapshot_dir/f"frame_{item.snapshot_id:06d}.npz")
        show_combined=item.snapshot_id==indices[len(indices)//2]
        controls=ViewerControls(show_combined=show_combined,show_stereo=not show_combined,show_tof=not show_combined)
        geometry=prepare_snapshot(item,config,controls);summary.append({"id":item.snapshot_id,"input":geometry.total_input_points,"displayed":geometry.total_displayed_points})
        if item.snapshot_id in indices:
            path=args.output_dir/f"preview_{item.snapshot_id:03d}.png";render_prepared_preview(geometry,path);previews.append(cv2.imread(str(path)))
    cv2.imwrite(str(args.output_dir/"sequence_preview.png"),np.vstack(previews))
    (args.output_dir/"summary.json").write_text(json.dumps({"frames":len(sequence),"snapshots":summary},indent=2)+"\n")
    print(f"frames={len(sequence)} first_displayed={summary[0]['displayed']} last_displayed={summary[-1]['displayed']}")


if __name__=="__main__":main()
