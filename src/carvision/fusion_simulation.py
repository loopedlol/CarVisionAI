"""Deterministic timestamped sensor generation and estimator/controller simulation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .control import (ControllerConfig, ControllerFeedback, ControllerState,
                      DifferentialDriveSimulator, Pose2D, TrajectoryCommand,
                      TrajectoryFollower, VehicleConfig, wrap_angle)
from .pose_estimation import (EncoderMeasurement, EstimatorConfig, EstimatorHealth,
                              EstimatorMode, EstimatorStatus, ExternalPoseMeasurement,
                              ImuYawRateMeasurement, PoseEstimator)


@dataclass(frozen=True)
class SensorSimulationConfig:
    encoder_left_scale: float = 1.0
    encoder_right_scale: float = 1.0
    imu_noise_std_rps: float = .025
    imu_bias_rps: float = .018
    external_position_noise_std_m: float = .035
    external_heading_noise_std_rad: float = .025
    external_interval_s: float = 1.0
    imu_delay_s: float = 0.0
    encoder_dropout: tuple[float, float] | None = None
    imu_dropout: tuple[float, float] | None = None
    external_dropout: tuple[float, float] | None = None
    external_outlier_time_s: float | None = None
    external_outlier_offset_m: float = 5.0

    def __post_init__(self) -> None:
        positive = (self.encoder_left_scale, self.encoder_right_scale,
                    self.imu_noise_std_rps, self.external_position_noise_std_m,
                    self.external_heading_noise_std_rad, self.external_interval_s,
                    self.external_outlier_offset_m)
        if min(positive) <= 0 or self.imu_delay_s < 0:
            raise ValueError("sensor scales, noise, intervals, and outlier size must be positive")


@dataclass(frozen=True)
class FusionMetrics:
    position_rmse_m: float
    maximum_position_error_m: float
    final_position_error_m: float
    heading_rmse_rad: float
    maximum_heading_error_rad: float
    within_three_sigma_fraction: float
    accepted_corrections: int
    rejected_corrections: int
    final_health: EstimatorHealth
    controller_state: ControllerState


@dataclass(frozen=True)
class FusionSimulationResult:
    mode: EstimatorMode
    feedback: tuple[ControllerFeedback, ...]
    estimator_status: tuple[EstimatorStatus, ...]
    truths: tuple[Pose2D, ...]
    imu_measurements_rps: tuple[float, ...]
    correction_events: tuple[tuple[int, bool], ...]
    encoder_dropout_mask: tuple[bool, ...]
    imu_dropout_mask: tuple[bool, ...]
    estimator_timings_ms: dict[str, tuple[float, ...]]
    runtime_ms: float
    metrics: FusionMetrics


def simulate_fused_trajectory(command: TrajectoryCommand, vehicle: VehicleConfig,
                              mode: EstimatorMode, *,
                              estimator_config: EstimatorConfig | None = None,
                              controller_config: ControllerConfig = ControllerConfig(),
                              sensors: SensorSimulationConfig = SensorSimulationConfig(),
                              initial_pose: Pose2D | None = None, seed: int = 0,
                              max_time_s: float = 45.0) -> FusionSimulationResult:
    first = command.trajectory.samples[0]
    pose = initial_pose or Pose2D(first.x_m, first.y_m, first.heading_rad)
    simulator = DifferentialDriveSimulator(vehicle, pose, seed)
    estimator = PoseEstimator(mode, estimator_config or EstimatorConfig(
        wheel_radius_m=vehicle.wheel_radius_m, track_width_m=vehicle.track_width_m))
    estimator.reset(pose, 0.0)
    follower = TrajectoryFollower(vehicle, controller_config)
    follower.accept(command, command.map_revision)
    rng = np.random.default_rng(seed + 1009)
    pending_imu: list[tuple[float, ImuYawRateMeasurement]] = []
    feedback: list[ControllerFeedback] = []
    statuses: list[EstimatorStatus] = []
    truths: list[Pose2D] = []
    imu_values: list[float] = []
    corrections: list[tuple[int, bool]] = []
    encoder_dropout, imu_dropout = [], []
    timings: dict[str, list[float]] = {"encoder": [], "imu": [], "external": []}
    next_external = sensors.external_interval_s
    outlier_sent = False
    begin = perf_counter()
    for step in range(int(max_time_s / vehicle.dt_s)):
        now = step * vehicle.dt_s
        item = follower.update(estimator.pose, now, current_map_revision=command.map_revision,
                               ground_truth_pose=simulator.pose,
                               wheel_saturated=simulator.saturated)
        truth, _, measured_wheels = simulator.step(item.commanded_wheels)
        sensor_time = now + vehicle.dt_s
        encoder_missing = _inside(sensor_time, sensors.encoder_dropout)
        imu_missing = _inside(sensor_time, sensors.imu_dropout)
        encoder_dropout.append(encoder_missing); imu_dropout.append(imu_missing)
        if not encoder_missing:
            measured_wheels = type(measured_wheels)(
                measured_wheels.left_rad_s * sensors.encoder_left_scale,
                measured_wheels.right_rad_s * sensors.encoder_right_scale)
            start = perf_counter()
            estimator.process_encoder(EncoderMeasurement(sensor_time, measured_wheels))
            timings["encoder"].append((perf_counter() - start) * 1000)
        else:
            estimator.note_encoder_dropout(sensor_time)
        imu_value = (simulator.actual_angular_rps + sensors.imu_bias_rps
                     + rng.normal(0, sensors.imu_noise_std_rps))
        imu_values.append(float(imu_value) if not imu_missing else np.nan)
        if not imu_missing and mode != EstimatorMode.ENCODER_ONLY:
            measurement = ImuYawRateMeasurement(sensor_time, float(imu_value),
                                                sensors.imu_noise_std_rps ** 2)
            pending_imu.append((sensor_time + sensors.imu_delay_s, measurement))
        due = [queued for queued in pending_imu if queued[0] <= sensor_time + 1e-12]
        pending_imu = [queued for queued in pending_imu if queued[0] > sensor_time + 1e-12]
        for _, measurement in due:
            start = perf_counter(); estimator.process_imu(measurement)
            timings["imu"].append((perf_counter() - start) * 1000)
        if (mode == EstimatorMode.ENCODER_IMU_EXTERNAL and sensor_time + 1e-12 >= next_external
                and not _inside(sensor_time, sensors.external_dropout)):
            observed = Pose2D(truth.x_m + rng.normal(0, sensors.external_position_noise_std_m),
                              truth.y_m + rng.normal(0, sensors.external_position_noise_std_m),
                              wrap_angle(truth.heading_rad
                                         + rng.normal(0, sensors.external_heading_noise_std_rad)))
            if (sensors.external_outlier_time_s is not None and not outlier_sent
                    and sensor_time >= sensors.external_outlier_time_s):
                observed = Pose2D(observed.x_m + sensors.external_outlier_offset_m,
                                  observed.y_m - sensors.external_outlier_offset_m,
                                  observed.heading_rad)
                outlier_sent = True
            start = perf_counter()
            accepted = estimator.process_external_pose(ExternalPoseMeasurement(sensor_time, observed))
            timings["external"].append((perf_counter() - start) * 1000)
            corrections.append((step, accepted)); next_external += sensors.external_interval_s
        estimator.advance_time(sensor_time)
        feedback.append(item); statuses.append(estimator.status()); truths.append(truth)
        if item.state in (ControllerState.STOPPED, ControllerState.FAULT,
                          ControllerState.EMERGENCY_STOP, ControllerState.WATCHDOG_STOP):
            break
    runtime = (perf_counter() - begin) * 1000
    metrics = _metrics(feedback, statuses, truths)
    return FusionSimulationResult(mode, tuple(feedback), tuple(statuses), tuple(truths),
                                  tuple(imu_values), tuple(corrections), tuple(encoder_dropout),
                                  tuple(imu_dropout), {key: tuple(value) for key, value in timings.items()},
                                  runtime, metrics)


def _metrics(feedback: list[ControllerFeedback], statuses: list[EstimatorStatus],
             truths: list[Pose2D]) -> FusionMetrics:
    position_errors, heading_errors, consistent = [], [], []
    for status, truth in zip(statuses, truths):
        dx, dy = status.pose.x_m - truth.x_m, status.pose.y_m - truth.y_m
        position_errors.append(float(np.hypot(dx, dy)))
        heading_errors.append(abs(wrap_angle(status.pose.heading_rad - truth.heading_rad)))
        residual = np.array([dx, dy, wrap_angle(status.pose.heading_rad - truth.heading_rad)])
        covariance = status.covariance[:3, :3]
        consistent.append(float(residual @ np.linalg.solve(covariance + np.eye(3) * 1e-12,
                                                            residual)) <= 9.0)
    corrections = statuses[-1].accepted_external if statuses else 0
    rejected = statuses[-1].rejected_outliers if statuses else 0
    return FusionMetrics(float(np.sqrt(np.mean(np.square(position_errors)))),
                         max(position_errors), position_errors[-1],
                         float(np.sqrt(np.mean(np.square(heading_errors)))),
                         max(heading_errors), float(np.mean(consistent)), corrections, rejected,
                         statuses[-1].health, feedback[-1].state)


def _inside(timestamp: float, window: tuple[float, float] | None) -> bool:
    return window is not None and window[0] <= timestamp < window[1]
