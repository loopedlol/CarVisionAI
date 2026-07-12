#!/usr/bin/env python3
"""Open static/live/replayed CarVision snapshots in the optional native viewer."""

import argparse
from pathlib import Path
from threading import Thread, Timer
from time import sleep

import numpy as np

from carvision.recorded_stereo import RecordedStereoAdapter, StereoRigCalibration
from carvision.visualization3d import (SCHEMA_VERSION, ViewerConfig, ViewerService,
    VisualizationSnapshot, SnapshotReplay, Open3DRenderer, run_open3d_smoke_test, save_snapshot)
from carvision.visualization_mock import make_mock_visualization_sequence


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest="command",required=True)
    static=sub.add_parser("static-mock");static.add_argument("--max-points",type=int,default=100000);add_debug_arguments(static)
    live=sub.add_parser("live-mock");live.add_argument("--frames",type=int,default=80);live.add_argument("--rate",type=float,default=12);live.add_argument("--save-folder",type=Path);add_debug_arguments(live)
    replay=sub.add_parser("replay");replay.add_argument("folder",type=Path);replay.add_argument("--speed",type=float,default=1.)
    recorded=sub.add_parser("recorded-pair");recorded.add_argument("--left",required=True);recorded.add_argument("--right",required=True);recorded.add_argument("--calibration",required=True);recorded.add_argument("--max-points",type=int,default=100000)
    disabled=sub.add_parser("disabled");disabled.add_argument("--frames",type=int,default=40)
    stats=sub.add_parser("stats");stats.add_argument("--frames",type=int,default=80)
    smoke=sub.add_parser("smoke-test");smoke.add_argument("--seconds",type=float,default=4.);smoke.add_argument("--screenshot",type=Path,default=Path("outputs/viewer/smoke_test.png"))
    args=parser.parse_args()
    if args.command=="smoke-test":
        if not run_open3d_smoke_test(args.screenshot,args.seconds):raise SystemExit("Open3D smoke test failed")
        print(f"smoke screenshot={args.screenshot}");return
    if args.command=="disabled":
        viewer=ViewerService(ViewerConfig(enabled=False));sequence=make_mock_visualization_sequence(args.frames)
        for item in sequence:viewer.publish(item)
        print(viewer.statistics());return
    if args.command=="recorded-pair":
        rig=StereoRigCalibration.load(args.calibration);_,_,result=RecordedStereoAdapter(rig).process_files(args.left,args.right)
        sequence=(VisualizationSnapshot(SCHEMA_VERSION,0,0,0,stereo_vehicle_points=result.points_vehicle_m,
            combined_vehicle_points=result.points_vehicle_m,sensor_origins_vehicle=np.asarray([rig.camera_to_vehicle.translation_m])),)
        config=ViewerConfig(max_points=args.max_points)
    elif args.command=="replay":
        source=SnapshotReplay.from_folder(args.folder,args.speed);sequence=tuple(load_all(source));config=ViewerConfig()
    else:
        sequence=make_mock_visualization_sequence(1 if args.command=="static-mock" else args.frames)
        config=ViewerConfig(max_points=getattr(args,"max_points",100000))
        if getattr(args,"save_folder",None):
            args.save_folder.mkdir(parents=True,exist_ok=True)
            for item in sequence:save_snapshot(item,args.save_folder/f"frame_{item.snapshot_id:06d}.npz")
    renderer=Open3DRenderer(diagnostics=getattr(args,"diagnostics",False),verification_screenshot=getattr(args,"verify_screenshot",None))
    viewer=ViewerService(config,renderer)
    rate=getattr(args,"rate",12.)*(getattr(args,"speed",1.))
    producer=Thread(target=publish_sequence,args=(viewer,sequence,rate,args.command=="replay"),daemon=True);producer.start()
    if getattr(args,"auto_close_seconds",None):Timer(args.auto_close_seconds,viewer.stop).start()
    try:viewer.run_foreground()
    except KeyboardInterrupt:viewer.stop()
    producer.join(timeout=.5);statistics=viewer.statistics();print(statistics)
    if statistics.rendered==0 and statistics.malformed:
        raise SystemExit("Native renderer did not open. Install Open3D using Python 3.10-3.12: pip install -e '.[visualization]'")


def load_all(replay):
    if not replay.paths:return
    yield replay.current()
    while replay.index<len(replay.paths)-1:yield replay.next()


def add_debug_arguments(parser):
    parser.add_argument("--diagnostics",action="store_true");parser.add_argument("--verify-screenshot",type=Path);parser.add_argument("--auto-close-seconds",type=float)


def publish_sequence(viewer,sequence,rate,replay_mode):
    index=0
    while index<len(sequence) and not viewer.statistics().closed:
        controls=viewer.controls
        if replay_mode:
            if controls.replay_restart_requested:index=0;controls.replay_restart_requested=False
            if controls.replay_previous_requested:index=max(0,index-1);controls.replay_previous_requested=False
            if controls.replay_next_requested:index=min(len(sequence)-1,index+1);controls.replay_next_requested=False
        if not controls.paused:
            viewer.publish(sequence[index]);index+=1
        sleep(1/max(rate,1e-6))


if __name__=="__main__":main()
