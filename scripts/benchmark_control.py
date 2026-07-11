#!/usr/bin/env python3
"""Benchmark closed-loop control separately from planning/trajectory generation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from carvision.control import TrajectoryCommand, VehicleConfig, simulate_trajectory
from carvision.control_mock import vehicle_scenario
from carvision.planning import PlannerConfig, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.trajectory import TrajectoryConfig, generate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/closed_loop/benchmark.json"))
    args = parser.parse_args()
    scenario = make_planning_scenario("blocked_direct")
    planner = PlannerConfig(robot_radius_m=.2, candidate_count=4)
    start = perf_counter()
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidate = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)[0]
    planning_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    trajectory = generate_trajectory(candidate, prepared, TrajectoryConfig(robot_radius_m=.2))
    trajectory_ms = (perf_counter() - start) * 1000
    command = TrajectoryCommand(1, 1, trajectory)
    data = {}
    for name in ("ideal", "moderate_slip", "severe"):
        runtime, updates, steps = [], [], []
        for iteration in range(args.iterations):
            result = simulate_trajectory(command, vehicle_scenario(name), seed=iteration,
                                         max_time_s=45)
            runtime.append(result.runtime_ms); updates.extend(result.controller_update_ms)
            steps.append(len(result.feedback))
        data[name] = {"simulation_median_ms": float(np.median(runtime)),
                      "controller_update_median_ms": float(np.median(updates)),
                      "controller_update_p99_ms": float(np.percentile(updates, 99)),
                      "steps_median": float(np.median(steps))}
    report = {"iterations": args.iterations, "planning_once_ms": planning_ms,
              "trajectory_generation_once_ms": trajectory_ms, "control": data}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
