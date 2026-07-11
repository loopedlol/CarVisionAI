#!/usr/bin/env python3
"""Run an existing planned trajectory under ideal, moderate, and severe dynamics."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from carvision.control import ControllerConfig, TrajectoryCommand, VehicleConfig, simulate_trajectory
from carvision.control_mock import vehicle_scenario
from carvision.control_visualization import render_simulation
from carvision.planning import PlannerConfig, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.trajectory import TrajectoryConfig, generate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/closed_loop"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_planning_scenario("blocked_direct")
    planner = PlannerConfig(robot_radius_m=.2, candidate_count=4)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    path = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)[0]
    trajectory = generate_trajectory(path, prepared,
                                     TrajectoryConfig(robot_radius_m=.2, safety_margin_m=.05,
                                                      sample_spacing_m=.04),
                                     final_heading_rad=np.pi / 2)
    command = TrajectoryCommand(1, 1, trajectory)
    base = VehicleConfig(control_frequency_hz=30, max_wheel_speed_rad_s=16)
    controller = ControllerConfig(saturation_fault_steps=18, terminal_timeout_s=4)
    report = {"trajectory_samples": len(trajectory.samples), "runs": {}}
    images = []
    for index, name in enumerate(("ideal", "moderate_slip", "severe"), 1):
        result = simulate_trajectory(command, vehicle_scenario(name, base), controller,
                                     seed=2026, max_time_s=45)
        final = result.feedback[-1]
        truth = final.ground_truth_pose
        estimate = final.estimated_pose
        report["runs"][name] = {
            "state": final.state.value, "fault": final.fault.value,
            "steps": len(result.feedback), "simulation_runtime_ms": result.runtime_ms,
            "controller_update_median_ms": float(np.median(result.controller_update_ms)),
            "controller_update_p99_ms": float(np.percentile(result.controller_update_ms, 99)),
            "final_truth": [truth.x_m, truth.y_m, truth.heading_rad],
            "final_estimate": [estimate.x_m, estimate.y_m, estimate.heading_rad],
            "final_position_error_m": final.position_error_m,
            "max_cross_track_error_m": max(f.cross_track_error_m for f in result.feedback),
            "max_heading_error_rad": max(abs(f.heading_error_rad) for f in result.feedback),
        }
        image = render_simulation(trajectory, result, f"{name} closed-loop trajectory following")
        cv2.imwrite(str(args.output_dir / f"0{index}_{name}.png"), image)
        images.append(cv2.resize(image, (750, 475)))
    cv2.imwrite(str(args.output_dir / "closed_loop_comparison.png"), np.vstack(images))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, item in report["runs"].items():
        print(f"{name}: state={item['state']} fault={item['fault']} "
              f"cross_track_max={item['max_cross_track_error_m']:.3f}m "
              f"controller_p50={item['controller_update_median_ms']:.3f}ms "
              f"simulation={item['simulation_runtime_ms']:.1f}ms")


if __name__ == "__main__": main()
