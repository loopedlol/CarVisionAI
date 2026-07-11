"""Closed-loop differential-drive trajectory control and deterministic simulation.

Coordinates follow the rest of CarVision: x points vehicle-right, y points
forward, and heading is zero along +y and positive toward +x.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import atan2, cos, pi, sin, sqrt
from time import perf_counter

import numpy as np

from .trajectory import GeneratedTrajectory, TrajectorySample


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + pi) % (2 * pi) - pi)


@dataclass(frozen=True)
class Pose2D:
    x_m: float
    y_m: float
    heading_rad: float


@dataclass(frozen=True)
class WheelSpeeds:
    left_rad_s: float
    right_rad_s: float


def body_to_wheels(linear_velocity_mps: float, angular_velocity_rps: float,
                   wheel_radius_m: float, track_width_m: float) -> WheelSpeeds:
    """Convert body velocity to wheels for positive heading toward vehicle-right."""
    if wheel_radius_m <= 0 or track_width_m <= 0:
        raise ValueError("wheel radius and track width must be positive")
    return WheelSpeeds(
        (linear_velocity_mps + angular_velocity_rps * track_width_m / 2) / wheel_radius_m,
        (linear_velocity_mps - angular_velocity_rps * track_width_m / 2) / wheel_radius_m,
    )


def wheels_to_body(wheels: WheelSpeeds, wheel_radius_m: float,
                   track_width_m: float) -> tuple[float, float]:
    if wheel_radius_m <= 0 or track_width_m <= 0:
        raise ValueError("wheel radius and track width must be positive")
    linear = wheel_radius_m * (wheels.left_rad_s + wheels.right_rad_s) / 2
    angular = wheel_radius_m * (wheels.left_rad_s - wheels.right_rad_s) / track_width_m
    return linear, angular


def integrate_unicycle(pose: Pose2D, linear_mps: float, angular_rps: float,
                       dt_s: float) -> Pose2D:
    """Exact constant-command integration in the project's heading convention."""
    if dt_s < 0:
        raise ValueError("dt_s cannot be negative")
    if abs(angular_rps) < 1e-10:
        return Pose2D(pose.x_m + linear_mps * sin(pose.heading_rad) * dt_s,
                      pose.y_m + linear_mps * cos(pose.heading_rad) * dt_s,
                      pose.heading_rad)
    new_heading = pose.heading_rad + angular_rps * dt_s
    radius = linear_mps / angular_rps
    return Pose2D(pose.x_m + radius * (cos(pose.heading_rad) - cos(new_heading)),
                  pose.y_m + radius * (sin(new_heading) - sin(pose.heading_rad)),
                  wrap_angle(new_heading))


@dataclass(frozen=True)
class VehicleConfig:
    wheel_radius_m: float = 0.065
    track_width_m: float = 0.32
    control_frequency_hz: float = 30.0
    max_linear_velocity_mps: float = 0.8
    max_angular_velocity_rps: float = 1.5
    max_linear_acceleration_mps2: float = 0.8
    max_angular_acceleration_rps2: float = 2.5
    max_wheel_speed_rad_s: float = 14.0
    command_delay_s: float = 0.0
    encoder_noise_std_rad_s: float = 0.0
    pose_position_noise_std_m: float = 0.0
    pose_heading_noise_std_rad: float = 0.0
    left_motor_scale: float = 1.0
    right_motor_scale: float = 1.0
    longitudinal_slip: float = 0.0
    angular_slip: float = 0.0
    motor_response_scale: float = 1.0

    def __post_init__(self) -> None:
        positive = (self.wheel_radius_m, self.track_width_m, self.control_frequency_hz,
                    self.max_linear_velocity_mps, self.max_angular_velocity_rps,
                    self.max_linear_acceleration_mps2, self.max_angular_acceleration_rps2,
                    self.max_wheel_speed_rad_s)
        if min(positive) <= 0:
            raise ValueError("vehicle dimensions, rates, and limits must be positive")
        if self.command_delay_s < 0 or min(self.left_motor_scale, self.right_motor_scale,
                                           self.motor_response_scale) < 0:
            raise ValueError("delay and motor scales cannot be negative")
        if not 0 <= self.longitudinal_slip <= 1 or not 0 <= self.angular_slip <= 1:
            raise ValueError("slip fractions must lie in [0,1]")

    @property
    def dt_s(self) -> float:
        return 1.0 / self.control_frequency_hz


@dataclass(frozen=True)
class ControllerConfig:
    lookahead_min_m: float = 0.18
    lookahead_speed_gain_s: float = 0.45
    position_tolerance_m: float = 0.06
    heading_tolerance_rad: float = 0.08
    cross_track_slow_m: float = 0.18
    max_cross_track_error_m: float = 0.55
    max_heading_error_rad: float = 1.35
    terminal_heading_gain: float = 2.0
    watchdog_timeout_s: float = 0.25
    saturation_fault_steps: int = 30
    terminal_timeout_s: float = 3.0


class ControllerState(str, Enum):
    IDLE = "idle"
    TRACKING = "tracking"
    TERMINAL_ALIGN = "terminal_align"
    STOPPED = "stopped"
    CANCELLED = "cancelled"
    FAULT = "fault"
    EMERGENCY_STOP = "emergency_stop"
    WATCHDOG_STOP = "watchdog_stop"


class FaultCode(str, Enum):
    NONE = "none"
    STALE_TRAJECTORY = "stale_trajectory"
    MAP_REVISION_MISMATCH = "map_revision_mismatch"
    CROSS_TRACK_ERROR = "excessive_cross_track_error"
    HEADING_ERROR = "excessive_heading_error"
    PERSISTENT_SATURATION = "persistent_wheel_saturation"
    TERMINAL_STOP_TIMEOUT = "terminal_stop_timeout"
    INVALIDATED = "trajectory_invalidated"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True)
class TrajectoryCommand:
    trajectory_id: int
    map_revision: int
    trajectory: GeneratedTrajectory


@dataclass(frozen=True)
class ControllerFeedback:
    timestamp_s: float
    trajectory_id: int | None
    map_revision: int | None
    progress_index: int
    progress_fraction: float
    accepted: bool
    watchdog_active: bool
    stopped: bool
    state: ControllerState
    fault: FaultCode
    commanded_linear_mps: float
    commanded_angular_rps: float
    commanded_wheels: WheelSpeeds
    estimated_pose: Pose2D
    ground_truth_pose: Pose2D | None
    position_error_m: float
    heading_error_rad: float
    cross_track_error_m: float
    wheel_saturated: bool


class TrajectoryFollower:
    """Pure-pursuit path tracking plus explicit terminal heading control."""

    def __init__(self, vehicle: VehicleConfig, config: ControllerConfig = ControllerConfig()) -> None:
        self.vehicle, self.config = vehicle, config
        self.active: TrajectoryCommand | None = None
        self.state, self.fault = ControllerState.IDLE, FaultCode.NONE
        self.progress_index = 0
        self.last_update_s: float | None = None
        self.last_v = self.last_w = 0.0
        self.saturation_steps = 0
        self.terminal_since_s: float | None = None

    def accept(self, command: TrajectoryCommand, current_map_revision: int,
               newest_trajectory_id: int | None = None) -> bool:
        if newest_trajectory_id is not None and command.trajectory_id < newest_trajectory_id:
            self._fault(FaultCode.STALE_TRAJECTORY)
            return False
        if command.map_revision != current_map_revision:
            self._fault(FaultCode.MAP_REVISION_MISMATCH)
            return False
        if not command.trajectory.samples:
            raise ValueError("trajectory has no samples")
        self.active, self.progress_index = command, 0
        self.state, self.fault = ControllerState.TRACKING, FaultCode.NONE
        self.last_v = self.last_w = 0.0
        self.saturation_steps = 0
        self.terminal_since_s = None
        return True

    def cancel(self, *, invalidated: bool = False) -> None:
        self.state = ControllerState.CANCELLED
        self.fault = FaultCode.INVALIDATED if invalidated else FaultCode.NONE
        self.last_v = self.last_w = 0.0

    def emergency_stop(self) -> None:
        self._fault(FaultCode.EMERGENCY_STOP, ControllerState.EMERGENCY_STOP)

    def watchdog(self, now_s: float, estimated_pose: Pose2D) -> ControllerFeedback | None:
        if self.last_update_s is not None and now_s - self.last_update_s > self.config.watchdog_timeout_s:
            self._fault(FaultCode.WATCHDOG_TIMEOUT, ControllerState.WATCHDOG_STOP)
            return self._feedback(now_s, estimated_pose, None, 0, 0, 0, False)
        return None

    def update(self, estimated_pose: Pose2D, now_s: float, *,
               current_map_revision: int, ground_truth_pose: Pose2D | None = None,
               wheel_saturated: bool = False) -> ControllerFeedback:
        if self.active is None or self.state not in (ControllerState.TRACKING,
                                                      ControllerState.TERMINAL_ALIGN):
            return self._feedback(now_s, estimated_pose, ground_truth_pose, 0, 0, 0, wheel_saturated)
        if current_map_revision != self.active.map_revision:
            self._fault(FaultCode.MAP_REVISION_MISMATCH)
            return self._feedback(now_s, estimated_pose, ground_truth_pose, 0, 0, 0, wheel_saturated)
        samples = self.active.trajectory.samples
        self.progress_index = self._nearest_progress(samples, estimated_pose)
        target_index = self._lookahead_index(samples, self.progress_index,
                                             self.config.lookahead_min_m
                                             + self.config.lookahead_speed_gain_s * abs(self.last_v))
        nearest, target, final = samples[self.progress_index], samples[target_index], samples[-1]
        cross_track = _distance(estimated_pose, nearest)
        path_heading_error = wrap_angle(nearest.heading_rad - estimated_pose.heading_rad)
        position_error = _distance(estimated_pose, final)
        final_heading_error = wrap_angle(final.heading_rad - estimated_pose.heading_rad)
        reported_heading_error = path_heading_error
        if cross_track > self.config.max_cross_track_error_m:
            self._fault(FaultCode.CROSS_TRACK_ERROR)
        elif (abs(path_heading_error) > self.config.max_heading_error_rad
              and cross_track > self.config.cross_track_slow_m):
            self._fault(FaultCode.HEADING_ERROR)
        if self.state == ControllerState.FAULT:
            return self._feedback(now_s, estimated_pose, ground_truth_pose,
                                  position_error, final_heading_error, cross_track, wheel_saturated)

        if position_error <= self.config.position_tolerance_m or self.state == ControllerState.TERMINAL_ALIGN:
            self.state = ControllerState.TERMINAL_ALIGN
            reported_heading_error = final_heading_error
            self.terminal_since_s = now_s if self.terminal_since_s is None else self.terminal_since_s
            desired_v = 0.0
            desired_w = np.clip(self.config.terminal_heading_gain * final_heading_error,
                                -self.vehicle.max_angular_velocity_rps,
                                self.vehicle.max_angular_velocity_rps)
            if abs(final_heading_error) <= self.config.heading_tolerance_rad:
                self.state = ControllerState.STOPPED
                desired_w = 0.0
            elif now_s - self.terminal_since_s > self.config.terminal_timeout_s:
                self._fault(FaultCode.TERMINAL_STOP_TIMEOUT)
                desired_w = 0.0
        else:
            dx, dy = target.x_m - estimated_pose.x_m, target.y_m - estimated_pose.y_m
            forward = dx * sin(estimated_pose.heading_rad) + dy * cos(estimated_pose.heading_rad)
            lateral = dx * cos(estimated_pose.heading_rad) - dy * sin(estimated_pose.heading_rad)
            distance2 = max(dx * dx + dy * dy, 1e-9)
            curvature = 2 * lateral / distance2
            # The lookahead sample may be the stationary terminal command while
            # the robot is still some distance away. Progress, rather than the
            # lookahead endpoint, owns the feed-forward speed.
            desired_v = min(nearest.linear_velocity_mps, self.vehicle.max_linear_velocity_mps)
            desired_v *= np.clip(1 - cross_track / self.config.max_cross_track_error_m, 0.15, 1.0)
            desired_v *= np.clip((forward + self.config.lookahead_min_m) /
                                 (2 * self.config.lookahead_min_m), 0.15, 1.0)
            desired_w = np.clip(curvature * desired_v,
                                -self.vehicle.max_angular_velocity_rps,
                                self.vehicle.max_angular_velocity_rps)
        dt = self.vehicle.dt_s if self.last_update_s is None else max(now_s - self.last_update_s, 1e-6)
        desired_v = _slew(self.last_v, float(desired_v), self.vehicle.max_linear_acceleration_mps2 * dt)
        desired_w = _slew(self.last_w, float(desired_w), self.vehicle.max_angular_acceleration_rps2 * dt)
        if self.state not in (ControllerState.TRACKING, ControllerState.TERMINAL_ALIGN):
            desired_v = desired_w = 0.0
        wheels = body_to_wheels(desired_v, desired_w, self.vehicle.wheel_radius_m,
                                self.vehicle.track_width_m)
        predicted_saturation = max(abs(wheels.left_rad_s), abs(wheels.right_rad_s)) > self.vehicle.max_wheel_speed_rad_s
        saturated = wheel_saturated or predicted_saturation
        self.saturation_steps = self.saturation_steps + 1 if saturated else 0
        if self.saturation_steps >= self.config.saturation_fault_steps:
            self._fault(FaultCode.PERSISTENT_SATURATION)
            desired_v = desired_w = 0.0
        self.last_v, self.last_w, self.last_update_s = desired_v, desired_w, now_s
        return self._feedback(now_s, estimated_pose, ground_truth_pose,
                              position_error, reported_heading_error, cross_track, saturated)

    def _nearest_progress(self, samples: tuple[TrajectorySample, ...], pose: Pose2D) -> int:
        stop = min(len(samples), self.progress_index + 80)
        distances = [(samples[i].x_m - pose.x_m) ** 2 + (samples[i].y_m - pose.y_m) ** 2
                     for i in range(self.progress_index, stop)]
        return self.progress_index + int(np.argmin(distances))

    @staticmethod
    def _lookahead_index(samples: tuple[TrajectorySample, ...], start: int, distance_m: float) -> int:
        target_distance = samples[start].distance_m + distance_m
        for index in range(start, len(samples)):
            if samples[index].distance_m >= target_distance:
                return index
        return len(samples) - 1

    def _fault(self, fault: FaultCode, state: ControllerState = ControllerState.FAULT) -> None:
        self.fault, self.state = fault, state
        self.last_v = self.last_w = 0.0

    def _feedback(self, now: float, pose: Pose2D, truth: Pose2D | None,
                  position_error: float, heading_error: float, cross_track: float,
                  saturated: bool) -> ControllerFeedback:
        wheels = body_to_wheels(self.last_v, self.last_w, self.vehicle.wheel_radius_m,
                                self.vehicle.track_width_m)
        count = len(self.active.trajectory.samples) if self.active else 1
        return ControllerFeedback(now, self.active.trajectory_id if self.active else None,
                                  self.active.map_revision if self.active else None,
                                  self.progress_index, self.progress_index / max(count - 1, 1),
                                  self.active is not None and self.fault == FaultCode.NONE,
                                  self.state == ControllerState.WATCHDOG_STOP,
                                  self.state == ControllerState.STOPPED,
                                  self.state, self.fault, self.last_v, self.last_w, wheels,
                                  pose, truth, position_error, heading_error, cross_track, saturated)


@dataclass
class DifferentialDriveSimulator:
    config: VehicleConfig
    pose: Pose2D
    seed: int = 0
    estimated_pose: Pose2D = field(init=False)
    actual_linear_mps: float = field(default=0.0, init=False)
    actual_angular_rps: float = field(default=0.0, init=False)
    saturated: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.estimated_pose = self.pose
        self._odometry_pose = self.pose
        self._rng = np.random.default_rng(self.seed)
        delay_steps = int(round(self.config.command_delay_s / self.config.dt_s))
        self._commands: deque[WheelSpeeds] = deque(
            [WheelSpeeds(0, 0)] * delay_steps, maxlen=delay_steps + 1)

    def step(self, commanded: WheelSpeeds) -> tuple[Pose2D, Pose2D, WheelSpeeds]:
        self._commands.append(commanded)
        delayed = self._commands.popleft() if len(self._commands) > 1 else self._commands[0]
        scaled = WheelSpeeds(delayed.left_rad_s * self.config.left_motor_scale * self.config.motor_response_scale,
                             delayed.right_rad_s * self.config.right_motor_scale * self.config.motor_response_scale)
        limit = self.config.max_wheel_speed_rad_s
        clipped = WheelSpeeds(float(np.clip(scaled.left_rad_s, -limit, limit)),
                              float(np.clip(scaled.right_rad_s, -limit, limit)))
        self.saturated = clipped != scaled
        v, w = wheels_to_body(clipped, self.config.wheel_radius_m, self.config.track_width_m)
        self.actual_linear_mps = v * (1 - self.config.longitudinal_slip)
        self.actual_angular_rps = w * (1 - self.config.angular_slip)
        self.pose = integrate_unicycle(self.pose, self.actual_linear_mps,
                                       self.actual_angular_rps, self.config.dt_s)
        measured = WheelSpeeds(
            clipped.left_rad_s + self._rng.normal(0, self.config.encoder_noise_std_rad_s),
            clipped.right_rad_s + self._rng.normal(0, self.config.encoder_noise_std_rad_s))
        odom_v, odom_w = wheels_to_body(measured, self.config.wheel_radius_m,
                                        self.config.track_width_m)
        self._odometry_pose = integrate_unicycle(self._odometry_pose, odom_v, odom_w,
                                                 self.config.dt_s)
        self.estimated_pose = Pose2D(
            self._odometry_pose.x_m + self._rng.normal(0, self.config.pose_position_noise_std_m),
            self._odometry_pose.y_m + self._rng.normal(0, self.config.pose_position_noise_std_m),
            wrap_angle(self._odometry_pose.heading_rad
                       + self._rng.normal(0, self.config.pose_heading_noise_std_rad)))
        return self.pose, self.estimated_pose, measured


@dataclass(frozen=True)
class SimulationResult:
    feedback: tuple[ControllerFeedback, ...]
    actual_linear_mps: tuple[float, ...]
    actual_angular_rps: tuple[float, ...]
    controller_update_ms: tuple[float, ...]
    runtime_ms: float


def simulate_trajectory(command: TrajectoryCommand, vehicle: VehicleConfig,
                        controller_config: ControllerConfig = ControllerConfig(), *,
                        initial_pose: Pose2D | None = None, seed: int = 0,
                        max_time_s: float = 40.0) -> SimulationResult:
    first = command.trajectory.samples[0]
    pose = initial_pose or Pose2D(first.x_m, first.y_m, first.heading_rad)
    simulator = DifferentialDriveSimulator(vehicle, pose, seed)
    follower = TrajectoryFollower(vehicle, controller_config)
    if not follower.accept(command, command.map_revision):
        raise ValueError("simulation command was rejected")
    feedback, actual_v, actual_w, timings = [], [], [], []
    begin = perf_counter()
    steps = int(max_time_s / vehicle.dt_s)
    for step in range(steps):
        now = step * vehicle.dt_s
        update_begin = perf_counter()
        item = follower.update(simulator.estimated_pose, now,
                               current_map_revision=command.map_revision,
                               ground_truth_pose=simulator.pose,
                               wheel_saturated=simulator.saturated)
        timings.append((perf_counter() - update_begin) * 1000)
        feedback.append(item)
        truth, estimate, _ = simulator.step(item.commanded_wheels)
        actual_v.append(simulator.actual_linear_mps)
        actual_w.append(simulator.actual_angular_rps)
        if item.state in (ControllerState.STOPPED, ControllerState.FAULT,
                          ControllerState.EMERGENCY_STOP, ControllerState.WATCHDOG_STOP):
            # One zero-command plant step has already been applied.
            break
    runtime = (perf_counter() - begin) * 1000
    return SimulationResult(tuple(feedback), tuple(actual_v), tuple(actual_w), tuple(timings), runtime)


def _distance(pose: Pose2D, sample: TrajectorySample) -> float:
    return sqrt((sample.x_m - pose.x_m) ** 2 + (sample.y_m - pose.y_m) ** 2)


def _slew(current: float, desired: float, maximum_delta: float) -> float:
    return float(np.clip(desired, current - maximum_delta, current + maximum_delta))
