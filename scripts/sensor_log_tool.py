#!/usr/bin/env python3
"""Recorded sensor dataset validation, calibration, replay, tuning, and reporting."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from carvision.pose_estimation import EstimatorMode
from carvision.recorded_log_mock import generate_fixture_dataset
from carvision.recorded_logs import (analyze_calibration, preprocess_dataset, replay_dataset,
                                    tune_parameters, validate_dataset, write_report)


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    fixture=sub.add_parser("generate-fixture");fixture.add_argument("directory",type=Path);fixture.add_argument("--run-type",default="straight")
    for name in ("validate","preprocess","analyze-imu","calibrate","replay","compare"):
        command=sub.add_parser(name);command.add_argument("dataset",type=Path)
        if name=="replay":command.add_argument("--mode",choices=[m.value for m in EstimatorMode],default="encoder_imu_external")
    tune=sub.add_parser("tune");tune.add_argument("--train",type=Path,nargs="+",required=True);tune.add_argument("--evaluate",type=Path,nargs="+",required=True);tune.add_argument("--output",type=Path,required=True)
    report=sub.add_parser("report");report.add_argument("datasets",type=Path,nargs="+");report.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="generate-fixture": result={"directory":str(generate_fixture_dataset(args.directory,run_type=args.run_type))}
    elif args.command=="validate": result=asdict(validate_dataset(args.dataset))
    elif args.command=="preprocess": result=asdict(preprocess_dataset(args.dataset))
    elif args.command in ("analyze-imu","calibrate"): result=asdict(analyze_calibration(args.dataset))
    elif args.command=="replay": result=asdict(replay_dataset(args.dataset,EstimatorMode(args.mode)).metrics)
    elif args.command=="compare": result={mode.value:asdict(replay_dataset(args.dataset,mode).metrics) for mode in EstimatorMode}
    elif args.command=="tune":
        result=tune_parameters(args.train,args.evaluate);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n")
    else: result=write_report(args.datasets,args.output_dir)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
