#!/usr/bin/env python3
"""Benchmark snapshot filtering, preparation, and non-blocking queue handoff."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from carvision.visualization3d import (SCHEMA_VERSION, HeadlessRenderer, SnapshotQueue,
    ViewerConfig, ViewerControls, ViewerService, VisualizationSnapshot,
    filter_and_downsample, prepare_snapshot)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--iterations",type=int,default=12);parser.add_argument("--output",type=Path,default=Path("outputs/viewer3d/benchmark.json"));args=parser.parse_args()
    rng=np.random.default_rng(19);report={}
    for count in (10000,50000,100000,300000):
        cloud=rng.uniform([-5,0,-1],[5,15,3],(count,3));cloud[::10003]=np.nan
        snapshot=VisualizationSnapshot(SCHEMA_VERSION,0,1,1,combined_vehicle_points=cloud,confidence=rng.random(count))
        filtering=[];preparation=[];handoff=[]
        queue=SnapshotQueue(2)
        for _ in range(args.iterations):
            start=perf_counter();filter_and_downsample(cloud,.035,100000,snapshot.confidence,.25);filtering.append((perf_counter()-start)*1000)
            start=perf_counter();prepare_snapshot(snapshot,ViewerConfig(max_points=100000),ViewerControls(show_combined=True));preparation.append((perf_counter()-start)*1000)
            start=perf_counter();queue.publish(snapshot);queue.newest();handoff.append((perf_counter()-start)*1000)
        report[str(count)]={"filter_median_ms":float(np.median(filtering)),"prepare_median_ms":float(np.median(preparation)),"queue_handoff_median_ms":float(np.median(handoff))}
    disabled=ViewerService(ViewerConfig(enabled=False),HeadlessRenderer());enabled=ViewerService(ViewerConfig(target_render_hz=500),HeadlessRenderer());enabled.start()
    sample=VisualizationSnapshot(SCHEMA_VERSION,0,1,1,combined_vehicle_points=np.zeros((1000,3)))
    start=perf_counter()
    for _ in range(10000):disabled.publish(sample)
    report["disabled_publish_10000_ms"]=(perf_counter()-start)*1000
    start=perf_counter()
    for _ in range(10000):enabled.publish(sample)
    report["enabled_publish_10000_ms"]=(perf_counter()-start)*1000;enabled.stop()
    report["enabled_dropped"]=enabled.statistics().dropped
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
