#!/usr/bin/env python3
"""Plan, generate, profile, update the map, and validate a trajectory."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from carvision.mapping import OCCUPIED, UNKNOWN
from carvision.planning import PlannerConfig, ScoreWeights, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.trajectory import TrajectoryConfig, generate_trajectory, validate_trajectory
from carvision.trajectory_visualization import render_trajectory
from carvision.trajectory_mock import make_trajectory_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trajectory_demo"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_planning_scenario("blocked_direct")
    planner = PlannerConfig(robot_radius_m=0.2, candidate_count=5,
                            weights=ScoreWeights(1, 0.45, 0.55, 1.5, 0.25))
    trajectory_config = TrajectoryConfig(robot_radius_m=0.2, safety_margin_m=0.05,
                                         sample_spacing_m=0.03, corner_radius_m=0.28)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    trajectory = generate_trajectory(candidates[0], prepared, trajectory_config)
    unchanged = validate_trajectory(trajectory, scenario.rough_data, scenario.config,
                                    planner, trajectory_config, current_speed_mps=0.6)

    unknown_map = scenario.rough_data.copy()
    middle = next(sample for sample in trajectory.samples
                  if sample.distance_m > 1.0
                  and prepared.data[_cell(sample.x_m, sample.y_m, scenario.config)] != UNKNOWN)
    middle_cell = _cell(middle.x_m, middle.y_m, scenario.config)
    unknown_map[middle_cell] = UNKNOWN
    unknown_validation = validate_trajectory(trajectory, unknown_map, scenario.config,
                                             planner, trajectory_config, current_speed_mps=0.4)
    unknown_prepared = prepare_planning_grid(unknown_map, scenario.config, planner=planner)

    obstacle_map = scenario.rough_data.copy()
    stop_distance = 0.35
    obstacle_sample = next(sample for sample in trajectory.samples if sample.distance_m >= stop_distance)
    obstacle_cell = _cell(obstacle_sample.x_m, obstacle_sample.y_m, scenario.config)
    obstacle_map[obstacle_cell] = OCCUPIED
    stop_validation = validate_trajectory(trajectory, obstacle_map, scenario.config,
                                          planner, trajectory_config, current_speed_mps=0.7)
    obstacle_prepared = prepare_planning_grid(obstacle_map, scenario.config, planner=planner)

    changed_unknown = unknown_map != scenario.rough_data
    changed_obstacle = obstacle_map != scenario.rough_data
    images = {
        "01_generated_trajectory.png": render_trajectory(prepared, trajectory, trajectory_config,
                                                         validation=unchanged),
        "02_unknown_speed_reduction.png": render_trajectory(
            unknown_prepared, trajectory, trajectory_config,
            validation=unknown_validation, changed_mask=changed_unknown),
        "03_braking_obstacle_stop.png": render_trajectory(
            obstacle_prepared, trajectory, trajectory_config,
            validation=stop_validation, changed_mask=changed_obstacle),
    }
    fallback_scenario = make_trajectory_scenario("unsafe_smoothing")
    fallback_planner = PlannerConfig(robot_radius_m=0.15, candidate_count=1, candidate_attempts=1)
    fallback_config = TrajectoryConfig(robot_radius_m=0.15, safety_margin_m=0.02,
                                       sample_spacing_m=0.03, corner_radius_m=0.4)
    fallback_prepared = prepare_planning_grid(fallback_scenario.data,
                                              fallback_scenario.config,
                                              planner=fallback_planner)
    fallback_trajectory = generate_trajectory(fallback_scenario.path,
                                              fallback_prepared, fallback_config)
    fallback_validation = validate_trajectory(
        fallback_trajectory, fallback_scenario.data, fallback_scenario.config,
        fallback_planner, fallback_config)
    cv2.imwrite(str(args.output_dir / "04_rejected_smoothing_fallback.png"),
                render_trajectory(fallback_prepared, fallback_trajectory,
                                  fallback_config, validation=fallback_validation))
    for name, image in images.items(): cv2.imwrite(str(args.output_dir / name), image)
    cv2.imwrite(str(args.output_dir / "trajectory_sequence.png"), np.hstack(tuple(images.values())))
    report = {
        "planner_candidate_count": len(candidates), "generation": asdict(trajectory),
        "unchanged_validation": asdict(unchanged),
        "unknown_update_validation": asdict(unknown_validation),
        "obstacle_update_validation": asdict(stop_validation),
        "fallback": {"used_fallback": fallback_trajectory.used_fallback,
                     "rejected_corners": fallback_trajectory.rejected_smoothing_corners},
    }
    # Numpy arrays from the geometry paths are summarized separately.
    report["generation"]["raw_metric_path"] = trajectory.raw_metric_path.tolist()
    report["generation"]["simplified_path"] = trajectory.simplified_path.tolist()
    report["generation"]["smoothed_path"] = trajectory.smoothed_path.tolist()
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"candidates={len(candidates)} samples={len(trajectory.samples)} "
          f"unknown_action={unknown_validation.action.value} "
          f"obstacle_action={stop_validation.action.value} "
          f"generation_ms={trajectory.timings_ms['total']:.2f}")


def _cell(x_m: float, y_m: float, config: object) -> tuple[int, int]:
    return (int((y_m - config.y_min_m) / config.resolution_m),
            int((x_m - config.x_min_m) / config.resolution_m))


if __name__ == "__main__": main()
