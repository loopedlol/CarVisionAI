#!/usr/bin/env python3
"""Compare timestamped pose estimators in moderate and severe closed loops."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from carvision.control import ControllerConfig, TrajectoryCommand, VehicleConfig
from carvision.control_mock import vehicle_scenario
from carvision.fusion_simulation import SensorSimulationConfig, simulate_fused_trajectory
from carvision.fusion_visualization import render_fusion_comparison
from carvision.planning import PlannerConfig, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.pose_estimation import EstimatorConfig, EstimatorMode
from carvision.trajectory import TrajectoryConfig, generate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pose_fusion"))
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_planning_scenario("blocked_direct")
    planner = PlannerConfig(robot_radius_m=.2, candidate_count=4)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidate = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)[0]
    trajectory = generate_trajectory(candidate, prepared,
                                     TrajectoryConfig(robot_radius_m=.2, safety_margin_m=.05,
                                                      sample_spacing_m=.04),
                                     final_heading_rad=np.pi / 2)
    command = TrajectoryCommand(2, 4, trajectory)
    base = VehicleConfig(control_frequency_hz=30, max_wheel_speed_rad_s=16)
    vehicle = vehicle_scenario("moderate_slip", base)
    estimator = EstimatorConfig(wheel_radius_m=vehicle.wheel_radius_m,
                                track_width_m=vehicle.track_width_m,
                                encoder_linear_noise_mps=.20,
                                encoder_angular_noise_rps=.12)
    sensors = SensorSimulationConfig(imu_bias_rps=.003, external_interval_s=.5,
                                     imu_delay_s=.06,
                                     imu_dropout=(8.0, 8.8), external_outlier_time_s=4.0)
    results = tuple(simulate_fused_trajectory(command, vehicle, mode,
                                               estimator_config=estimator, sensors=sensors,
                                               seed=2026, max_time_s=45)
                    for mode in EstimatorMode)
    cv2.imwrite(str(args.output_dir / "moderate_fusion_comparison.png"),
                render_fusion_comparison(results, "Moderate slip: timestamped pose fusion"))

    severe_vehicle = vehicle_scenario("severe", base)
    severe_estimator = EstimatorConfig(
        wheel_radius_m=severe_vehicle.wheel_radius_m, track_width_m=severe_vehicle.track_width_m,
        encoder_linear_noise_mps=.20, degraded_position_std_m=.025,
        fault_position_std_m=.06, degraded_heading_std_rad=.08)
    severe_sensors = SensorSimulationConfig(encoder_dropout=(.5, 10), imu_dropout=(.5, 10),
                                            external_dropout=(0, 10), external_interval_s=.5)
    severe = simulate_fused_trajectory(command, severe_vehicle,
                                       EstimatorMode.ENCODER_IMU_EXTERNAL,
                                       estimator_config=severe_estimator,
                                       controller_config=ControllerConfig(saturation_fault_steps=18),
                                       sensors=severe_sensors, seed=2026, max_time_s=30)
    cv2.imwrite(str(args.output_dir / "severe_degraded_estimator.png"),
                render_fusion_comparison((severe,), "Severe disturbance and sensor dropout"))
    report = {"moderate": {result.mode.value: _summary(result) for result in results},
              "severe": _summary(severe)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, result in [(r.mode.value, r) for r in results] + [("severe", severe)]:
        metric = result.metrics
        print(f"{name}: position_rmse={metric.position_rmse_m:.3f}m "
              f"final={metric.final_position_error_m:.3f}m heading_rmse={metric.heading_rmse_rad:.3f}rad "
              f"health={metric.final_health.value} controller={metric.controller_state.value}")


def _summary(result):
    metric = asdict(result.metrics)
    metric["final_health"] = result.metrics.final_health.value
    metric["controller_state"] = result.metrics.controller_state.value
    metric["steps"] = len(result.feedback); metric["runtime_ms"] = result.runtime_ms
    metric["update_median_ms"] = {key: float(np.median(value)) if value else None
                                  for key, value in result.estimator_timings_ms.items()}
    metric["update_p99_ms"] = {key: float(np.percentile(value, 99)) if value else None
                               for key, value in result.estimator_timings_ms.items()}
    return metric


if __name__ == "__main__": main()
