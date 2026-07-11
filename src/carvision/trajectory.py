"""Conservative geometric trajectories for a circular differential-drive robot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, pi, sqrt
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .map_events import PlanningAction
from .mapping import OCCUPIED, UNKNOWN, OccupancyGridConfig
from .planning import Cell, PathCandidate, PlannerConfig, PreparedPlanningGrid, cell_to_metric, prepare_planning_grid


@dataclass(frozen=True)
class TrajectoryConfig:
    robot_radius_m: float = 0.20
    safety_margin_m: float = 0.08
    sample_spacing_m: float = 0.04
    corner_radius_m: float = 0.25
    smoothing_enabled: bool = True
    max_linear_velocity_mps: float = 0.8
    max_angular_velocity_rps: float = 1.5
    max_linear_acceleration_mps2: float = 0.6
    max_linear_deceleration_mps2: float = 0.8
    max_angular_acceleration_rps2: float = 2.0
    max_angular_deceleration_rps2: float = 2.5
    reaction_latency_s: float = 0.20
    braking_safety_margin_m: float = 0.10
    clearance_slow_distance_m: float = 0.65
    unknown_speed_factor: float = 0.40
    minimum_motion_speed_mps: float = 0.04

    def __post_init__(self) -> None:
        positive = (self.sample_spacing_m, self.max_linear_velocity_mps,
                    self.max_angular_velocity_rps, self.max_linear_acceleration_mps2,
                    self.max_linear_deceleration_mps2, self.max_angular_acceleration_rps2,
                    self.max_angular_deceleration_rps2, self.clearance_slow_distance_m)
        if min(positive) <= 0:
            raise ValueError("trajectory sampling, limits, and clearance distance must be positive")
        if min(self.robot_radius_m, self.safety_margin_m, self.corner_radius_m,
               self.reaction_latency_s, self.braking_safety_margin_m) < 0:
            raise ValueError("geometry and latency settings must be non-negative")
        if not 0 < self.unknown_speed_factor <= 1:
            raise ValueError("unknown_speed_factor must lie in (0,1]")

    @property
    def required_clearance_m(self) -> float:
        return self.robot_radius_m + self.safety_margin_m


@dataclass(frozen=True)
class TrajectorySample:
    x_m: float
    y_m: float
    heading_rad: float
    curvature_per_m: float
    clearance_m: float
    unknown: bool
    linear_velocity_mps: float
    angular_velocity_rps: float
    time_s: float
    distance_m: float


@dataclass(frozen=True)
class GeneratedTrajectory:
    raw_metric_path: NDArray[np.float64]
    simplified_path: NDArray[np.float64]
    smoothed_path: NDArray[np.float64]
    samples: tuple[TrajectorySample, ...]
    used_smoothing: bool
    used_fallback: bool
    rejected_smoothing_corners: tuple[int, ...]
    timings_ms: dict[str, float]


class TrajectoryAction(str, Enum):
    VALID = "trajectory_valid"
    REDUCE_SPEED = "reduce_trajectory_speed"
    REGENERATE = "regenerate_trajectory_same_path"
    REPLAN = "fast_path_replan_required"
    STOP = "immediate_stop_required"


@dataclass(frozen=True)
class TrajectoryValidation:
    action: TrajectoryAction
    reasons: tuple[str, ...]
    trajectory_valid: bool
    grid_path_valid: bool
    collision_sample_indices: tuple[int, ...]
    stopping_corridor_collision_indices: tuple[int, ...]
    recommended_speed_scale: float
    braking_distance_m: float
    validation_time_ms: float


def grid_path_to_metric(path: tuple[Cell, ...], config: OccupancyGridConfig) -> NDArray[np.float64]:
    if not path:
        raise ValueError("grid path cannot be empty")
    return np.asarray([cell_to_metric(cell, config) for cell in path], dtype=np.float64)


def remove_duplicate_and_collinear(points: NDArray[np.floating], *,
                                   tolerance: float = 1e-9) -> NDArray[np.float64]:
    values = _validate_points(points)
    kept = [values[0]]
    for point in values[1:]:
        if np.linalg.norm(point - kept[-1]) > tolerance:
            kept.append(point)
    if len(kept) <= 2:
        return np.asarray(kept)
    output = [kept[0]]
    for index in range(1, len(kept) - 1):
        first, middle, last = output[-1], kept[index], kept[index + 1]
        first_vector, second_vector = middle - first, last - middle
        cross = abs(first_vector[0] * second_vector[1]
                    - first_vector[1] * second_vector[0])
        if cross <= tolerance and np.dot(middle - first, last - middle) >= 0:
            continue
        output.append(middle)
    output.append(kept[-1])
    return np.asarray(output, dtype=np.float64)


def segment_is_collision_free(start: NDArray[np.floating], end: NDArray[np.floating],
                              prepared: PreparedPlanningGrid,
                              config: TrajectoryConfig) -> bool:
    return not collision_indices(sample_polyline(np.asarray([start, end]), config.sample_spacing_m),
                                 prepared, config)


def simplify_line_of_sight(points: NDArray[np.floating], prepared: PreparedPlanningGrid,
                           config: TrajectoryConfig) -> NDArray[np.float64]:
    values = remove_duplicate_and_collinear(points)
    if len(values) <= 2:
        return values
    simplified = [values[0]]
    anchor = 0
    while anchor < len(values) - 1:
        selected = anchor + 1
        for candidate in range(len(values) - 1, anchor, -1):
            if segment_is_collision_free(values[anchor], values[candidate], prepared, config):
                selected = candidate
                break
        simplified.append(values[selected])
        anchor = selected
    return np.asarray(simplified)


def smooth_corners(points: NDArray[np.floating], prepared: PreparedPlanningGrid,
                   config: TrajectoryConfig) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    values = _validate_points(points)
    if not config.smoothing_enabled or config.corner_radius_m <= 0 or len(values) < 3:
        return values.copy(), ()
    output: list[np.ndarray] = [values[0]]
    rejected: list[int] = []
    for index in range(1, len(values) - 1):
        previous, corner, following = values[index - 1:index + 2]
        incoming = corner - previous
        outgoing = following - corner
        incoming_length, outgoing_length = np.linalg.norm(incoming), np.linalg.norm(outgoing)
        if incoming_length < 1e-9 or outgoing_length < 1e-9:
            continue
        distance = min(config.corner_radius_m, 0.35 * incoming_length, 0.35 * outgoing_length)
        entry = corner - incoming / incoming_length * distance
        exit_point = corner + outgoing / outgoing_length * distance
        count = max(3, int(np.ceil(2 * distance / config.sample_spacing_m)) + 1)
        parameter = np.linspace(0, 1, count)
        curve = ((1 - parameter)[:, None] ** 2 * entry
                 + 2 * (1 - parameter)[:, None] * parameter[:, None] * corner
                 + parameter[:, None] ** 2 * exit_point)
        trial = np.vstack((output[-1], curve, following))
        if collision_indices(sample_polyline(trial, config.sample_spacing_m), prepared, config):
            rejected.append(index)
            output.append(corner)
        else:
            if np.linalg.norm(output[-1] - entry) > 1e-9:
                output.append(entry)
            output.extend(curve[1:])
    output.append(values[-1])
    return remove_duplicate_and_collinear(np.asarray(output), tolerance=1e-10), tuple(rejected)


def sample_polyline(points: NDArray[np.floating], spacing_m: float) -> NDArray[np.float64]:
    values = _validate_points(points)
    if spacing_m <= 0:
        raise ValueError("spacing must be positive")
    samples = [values[0]]
    for start, end in zip(values, values[1:]):
        length = np.linalg.norm(end - start)
        if length <= 1e-12:
            continue
        count = max(1, int(np.ceil(length / spacing_m)))
        for index in range(1, count + 1):
            samples.append(start + (end - start) * (index / count))
    return np.asarray(samples)


def collision_indices(points: NDArray[np.floating], prepared: PreparedPlanningGrid,
                      config: TrajectoryConfig) -> tuple[int, ...]:
    values = _validate_points(points)
    indices = []
    for index, (x_m, y_m) in enumerate(values):
        cell = _metric_cell_or_none(float(x_m), float(y_m), prepared.config)
        if cell is None or prepared.clearance_m[cell] < config.required_clearance_m - 1e-6:
            indices.append(index)
            continue
        if prepared.data[cell] == UNKNOWN and not bool(prepared.traversable[cell]):
            indices.append(index)
    return tuple(indices)


def generate_trajectory(candidate: PathCandidate | tuple[Cell, ...],
                        prepared: PreparedPlanningGrid,
                        config: TrajectoryConfig = TrajectoryConfig(), *,
                        final_heading_rad: float | None = None) -> GeneratedTrajectory:
    timings: dict[str, float] = {}
    path = candidate.cells if isinstance(candidate, PathCandidate) else candidate
    begin = perf_counter()
    raw = grid_path_to_metric(path, prepared.config)
    cleaned = remove_duplicate_and_collinear(raw)
    simplified = simplify_line_of_sight(cleaned, prepared, config)
    timings["simplification"] = (perf_counter() - begin) * 1000
    begin = perf_counter()
    smoothed, rejected = smooth_corners(simplified, prepared, config)
    used_smoothing = len(smoothed) > len(simplified) and not rejected
    timings["smoothing"] = (perf_counter() - begin) * 1000
    begin = perf_counter()
    dense = sample_polyline(smoothed, min(config.sample_spacing_m,
                                           prepared.config.resolution_m * 0.5,
                                           max(config.robot_radius_m * 0.25, 0.01)))
    collisions = collision_indices(dense, prepared, config)
    used_fallback = bool(rejected) or bool(collisions)
    if used_fallback:
        smoothed = simplified.copy()
        dense = sample_polyline(smoothed, min(config.sample_spacing_m,
                                               prepared.config.resolution_m * 0.5))
        if collision_indices(dense, prepared, config):
            # Last conservative fallback retains the raw grid path. A collision
            # here is a genuine path/map integration error.
            smoothed = raw.copy()
            dense = sample_polyline(smoothed, min(config.sample_spacing_m,
                                                   prepared.config.resolution_m * 0.5))
            if collision_indices(dense, prepared, config):
                raise ValueError("raw grid path is not swept-footprint collision free")
    timings["collision_checking"] = (perf_counter() - begin) * 1000
    begin = perf_counter()
    samples = _profile_samples(dense, prepared, config, final_heading_rad)
    timings["speed_profile"] = (perf_counter() - begin) * 1000
    timings["total"] = sum(timings.values())
    return GeneratedTrajectory(raw, simplified, smoothed, samples, used_smoothing,
                               used_fallback, rejected, timings)


def stopping_distance_m(speed_mps: float, config: TrajectoryConfig) -> float:
    if speed_mps < 0:
        raise ValueError("speed cannot be negative")
    return (speed_mps * config.reaction_latency_s
            + speed_mps ** 2 / (2 * config.max_linear_deceleration_mps2)
            + config.braking_safety_margin_m)


def validate_trajectory(trajectory: GeneratedTrajectory, stable_map: NDArray[np.integer],
                        map_config: OccupancyGridConfig, planner: PlannerConfig,
                        config: TrajectoryConfig, *, current_index: int = 0,
                        current_speed_mps: float | None = None) -> TrajectoryValidation:
    begin = perf_counter()
    if not 0 <= current_index < len(trajectory.samples):
        raise ValueError("current_index outside trajectory")
    prepared = prepare_planning_grid(stable_map, map_config, planner=planner)
    points = np.asarray([(sample.x_m, sample.y_m) for sample in trajectory.samples])
    collisions = collision_indices(points, prepared, config)
    raw_points = trajectory.raw_metric_path
    grid_path_valid = not collision_indices(sample_polyline(raw_points, min(config.sample_spacing_m,
                                                                            map_config.resolution_m * 0.5)),
                                            prepared, config)
    speed = (trajectory.samples[current_index].linear_velocity_mps
             if current_speed_mps is None else current_speed_mps)
    braking = stopping_distance_m(speed, config)
    current_distance = trajectory.samples[current_index].distance_m
    stop_indices = tuple(index for index in collisions if index >= current_index
                         and trajectory.samples[index].distance_m - current_distance <= braking + 1e-9)
    reasons: list[str] = []
    action = TrajectoryAction.VALID
    if stop_indices:
        action = TrajectoryAction.STOP
        reasons.append(f"obstacle intersects swept footprint within {braking:.2f} m stopping corridor")
    elif collisions and grid_path_valid:
        action = TrajectoryAction.REGENERATE
        reasons.append("smoothed trajectory unsafe while selected grid path remains valid")
    elif collisions:
        action = TrajectoryAction.REPLAN
        reasons.append("updated map invalidates both trajectory and selected grid path")
    else:
        recommended = _recommended_change_scale(trajectory.samples[current_index:], prepared, config)
        if recommended < 0.85:
            action = TrajectoryAction.REDUCE_SPEED
            reasons.append(f"clearance or unknown exposure recommends speed scale {recommended:.2f}")
        else:
            reasons.append("swept footprint and speed-dependent stopping corridor remain valid")
    recommended = _recommended_change_scale(trajectory.samples[current_index:], prepared, config)
    return TrajectoryValidation(action, tuple(reasons), not collisions, grid_path_valid,
                                collisions, stop_indices, recommended, braking,
                                (perf_counter() - begin) * 1000)


def combine_event_and_trajectory_action(event_action: PlanningAction,
                                        validation: TrajectoryValidation) -> TrajectoryAction:
    if event_action == PlanningAction.STOP or validation.action == TrajectoryAction.STOP:
        return TrajectoryAction.STOP
    if event_action in (PlanningAction.FAST_REPLAN, PlanningAction.FULL_REPLAN):
        return TrajectoryAction.REPLAN
    if validation.action in (TrajectoryAction.REPLAN, TrajectoryAction.REGENERATE,
                             TrajectoryAction.REDUCE_SPEED):
        return validation.action
    if event_action == PlanningAction.REEVALUATE:
        return TrajectoryAction.REPLAN
    return TrajectoryAction.VALID


def _profile_samples(points: NDArray[np.float64], prepared: PreparedPlanningGrid,
                     config: TrajectoryConfig, final_heading: float | None) -> tuple[TrajectorySample, ...]:
    if len(points) < 2:
        raise ValueError("trajectory needs at least two distinct points")
    delta = np.diff(points, axis=0)
    segment = np.linalg.norm(delta, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    headings = np.empty(len(points), dtype=float)
    headings[:-1] = np.arctan2(delta[:, 0], delta[:, 1])
    headings[-1] = headings[-2]
    headings = np.unwrap(headings)
    curvature = np.zeros(len(points), dtype=float)
    if len(points) > 2:
        heading_delta = np.diff(headings)
        curvature[1:] = heading_delta / np.maximum(segment, 1e-9)
        curvature[0] = curvature[1]
    clearance = np.empty(len(points), dtype=float)
    unknown = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        cell = _metric_cell_or_none(point[0], point[1], prepared.config)
        if cell is None:
            raise ValueError("sample outside map")
        clearance[index] = prepared.clearance_m[cell]
        unknown[index] = prepared.data[cell] == UNKNOWN
    velocity = np.full(len(points), config.max_linear_velocity_mps)
    turn_limit = np.full(len(curvature), np.inf)
    np.divide(config.max_angular_velocity_rps, np.abs(curvature), out=turn_limit,
              where=np.abs(curvature) > 1e-9)
    velocity = np.minimum(velocity, turn_limit)
    clearance_factor = np.clip(
        (clearance - config.required_clearance_m) / config.clearance_slow_distance_m,
        config.minimum_motion_speed_mps / config.max_linear_velocity_mps, 1.0)
    velocity *= clearance_factor
    velocity[unknown] *= config.unknown_speed_factor
    remaining = cumulative[-1] - cumulative
    velocity = np.minimum(velocity, np.sqrt(2 * config.max_linear_deceleration_mps2
                                            * np.maximum(remaining, 0)))
    velocity[0] = min(velocity[0], config.minimum_motion_speed_mps)
    velocity[-1] = 0.0
    for index in range(1, len(velocity)):
        velocity[index] = min(velocity[index], sqrt(max(
            0.0, velocity[index - 1] ** 2
            + 2 * config.max_linear_acceleration_mps2 * segment[index - 1])))
    for index in range(len(velocity) - 2, -1, -1):
        velocity[index] = min(velocity[index], sqrt(max(
            0.0, velocity[index + 1] ** 2
            + 2 * config.max_linear_deceleration_mps2 * segment[index])))
    angular = np.clip(curvature * velocity, -config.max_angular_velocity_rps,
                      config.max_angular_velocity_rps)
    times = np.zeros(len(points), dtype=float)
    for index in range(1, len(points)):
        average = (velocity[index - 1] + velocity[index]) / 2
        if average > 1e-6:
            dt = segment[index - 1] / average
        else:
            dt = sqrt(2 * segment[index - 1] / config.max_linear_acceleration_mps2)
        dt = max(dt, 1e-4)
        max_delta = (config.max_angular_acceleration_rps2 if abs(angular[index]) >= abs(angular[index - 1])
                     else config.max_angular_deceleration_rps2) * dt
        angular[index] = np.clip(angular[index], angular[index - 1] - max_delta,
                                 angular[index - 1] + max_delta)
        times[index] = times[index - 1] + dt
    samples = [TrajectorySample(
        float(point[0]), float(point[1]), float(_wrap_angle(headings[index])),
        float(curvature[index]), float(clearance[index]), bool(unknown[index]),
        float(velocity[index]), float(angular[index]), float(times[index]),
        float(cumulative[index])) for index, point in enumerate(points)]
    if final_heading is not None:
        desired = _wrap_angle(final_heading)
        difference = _angle_difference(samples[-1].heading_rad, desired)
        if abs(difference) > 1e-6:
            last = samples[-1]
            sign = 1.0 if difference > 0 else -1.0
            angle = abs(difference)
            acceleration = config.max_angular_acceleration_rps2
            deceleration = config.max_angular_deceleration_rps2
            triangular_peak = sqrt(2 * angle / (1 / acceleration + 1 / deceleration))
            peak = min(config.max_angular_velocity_rps, triangular_peak)
            acceleration_time = peak / acceleration
            deceleration_time = peak / deceleration
            acceleration_angle = 0.5 * peak * acceleration_time
            deceleration_angle = 0.5 * peak * deceleration_time
            cruise_angle = max(0.0, angle - acceleration_angle - deceleration_angle)
            acceleration_heading = _wrap_angle(last.heading_rad + sign * acceleration_angle)
            samples.append(TrajectorySample(
                last.x_m, last.y_m, acceleration_heading, 0.0, last.clearance_m,
                last.unknown, 0.0, sign * peak, last.time_s + acceleration_time,
                last.distance_m))
            if cruise_angle > 1e-9:
                cruise_time = cruise_angle / peak
                cruise_heading = _wrap_angle(acceleration_heading + sign * cruise_angle)
                samples.append(TrajectorySample(
                    last.x_m, last.y_m, cruise_heading, 0.0, last.clearance_m,
                    last.unknown, 0.0, sign * peak,
                    samples[-1].time_s + cruise_time, last.distance_m))
            samples.append(TrajectorySample(
                last.x_m, last.y_m, desired, 0.0, last.clearance_m, last.unknown,
                0.0, 0.0, samples[-1].time_s + deceleration_time, last.distance_m))
    # The executable terminal command must be stationary.
    last = samples[-1]
    samples[-1] = TrajectorySample(last.x_m, last.y_m, last.heading_rad, last.curvature_per_m,
                                   last.clearance_m, last.unknown, 0.0, 0.0,
                                   last.time_s, last.distance_m)
    return tuple(samples)


def _recommended_scale(points: NDArray[np.float64], prepared: PreparedPlanningGrid,
                       config: TrajectoryConfig) -> float:
    scale = 1.0
    for point in points:
        cell = _metric_cell_or_none(point[0], point[1], prepared.config)
        if cell is None:
            return 0.0
        clearance_scale = np.clip(
            (prepared.clearance_m[cell] - config.required_clearance_m)
            / config.clearance_slow_distance_m, 0.0, 1.0)
        scale = min(scale, float(clearance_scale))
        if prepared.data[cell] == UNKNOWN:
            scale = min(scale, config.unknown_speed_factor)
    return scale


def _recommended_change_scale(samples: tuple[TrajectorySample, ...],
                              prepared: PreparedPlanningGrid,
                              config: TrajectoryConfig) -> float:
    """Return speed scale relative to conditions used for the existing profile."""
    scale = 1.0
    for sample in samples:
        cell = _metric_cell_or_none(sample.x_m, sample.y_m, prepared.config)
        if cell is None:
            return 0.0
        new_clearance = prepared.clearance_m[cell]
        if new_clearance + 1e-6 < sample.clearance_m:
            old_factor = np.clip((sample.clearance_m - config.required_clearance_m)
                                 / config.clearance_slow_distance_m, 1e-3, 1.0)
            new_factor = np.clip((new_clearance - config.required_clearance_m)
                                 / config.clearance_slow_distance_m, 0.0, 1.0)
            scale = min(scale, float(new_factor / old_factor))
        if prepared.data[cell] == UNKNOWN and not sample.unknown:
            scale = min(scale, config.unknown_speed_factor)
    return scale


def _metric_cell_or_none(x_m: float, y_m: float,
                         config: OccupancyGridConfig) -> Cell | None:
    if not (config.x_min_m <= x_m < config.x_max_m
            and config.y_min_m <= y_m < config.y_max_m):
        return None
    column = min(int(np.floor((x_m - config.x_min_m) / config.resolution_m)), config.width - 1)
    row = min(int(np.floor((y_m - config.y_min_m) / config.resolution_m)), config.height - 1)
    return row, column


def _validate_points(points: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("points must be a finite Nx2 array")
    return values


def _wrap_angle(angle: float) -> float:
    return float((angle + pi) % (2 * pi) - pi)


def _angle_difference(source: float, target: float) -> float:
    return _wrap_angle(target - source)
