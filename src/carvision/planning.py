"""Sensor-independent local grid planning and directed map refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from math import atan2, pi
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, OccupancyGridConfig

Cell = tuple[int, int]  # row, column


@dataclass(frozen=True)
class PlanningPose:
    """Vehicle-frame local pose; heading zero is forward (+y), positive toward +x."""

    x_m: float
    y_m: float
    heading_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite((self.x_m, self.y_m, self.heading_rad)).all():
            raise ValueError("planning pose must be finite")


@dataclass(frozen=True)
class ScoreWeights:
    length: float = 1.0
    clearance: float = 0.8
    unknown: float = 3.0
    narrow: float = 2.0
    heading_change: float = 0.35

    def __post_init__(self) -> None:
        if min(self.length, self.clearance, self.unknown, self.narrow,
               self.heading_change) < 0:
            raise ValueError("score weights must be non-negative")


@dataclass(frozen=True)
class PlannerConfig:
    robot_radius_m: float = 0.25
    unknown_policy: Literal["block", "penalize", "allow"] = "penalize"
    narrow_clearance_m: float = 0.45
    candidate_count: int = 5
    candidate_attempts: int = 10
    diversity_radius_m: float = 0.35
    diversity_penalty: float = 8.0
    max_path_overlap: float = 0.78
    inspection_radius_m: float = 0.6
    weights: ScoreWeights = ScoreWeights()

    def __post_init__(self) -> None:
        if self.robot_radius_m < 0 or self.narrow_clearance_m <= 0:
            raise ValueError("robot radius must be non-negative and narrow clearance positive")
        if self.unknown_policy not in ("block", "penalize", "allow"):
            raise ValueError("unknown_policy must be block, penalize, or allow")
        if self.candidate_count < 1 or self.candidate_attempts < self.candidate_count:
            raise ValueError("candidate counts are inconsistent")
        if self.diversity_radius_m < 0 or self.diversity_penalty < 0:
            raise ValueError("diversity settings must be non-negative")
        if not 0 <= self.max_path_overlap <= 1 or self.inspection_radius_m <= 0:
            raise ValueError("overlap must be [0,1] and inspection radius positive")


@dataclass(frozen=True)
class PreparedPlanningGrid:
    data: NDArray[np.int8]
    config: OccupancyGridConfig
    inflated_unsafe: NDArray[np.bool_]
    traversable: NDArray[np.bool_]
    clearance_m: NDArray[np.float64]


@dataclass(frozen=True)
class PathScore:
    length_m: float
    clearance_cost: float
    minimum_clearance_m: float
    mean_clearance_m: float
    unknown_distance_m: float
    unknown_fraction: float
    narrow_distance_m: float
    narrow_fraction: float
    heading_change_rad: float
    traversal_difficulty: float
    total: float


@dataclass(frozen=True)
class PathCandidate:
    cells: tuple[Cell, ...]
    score: PathScore
    rank: int = 0


@dataclass(frozen=True)
class InspectionTarget:
    center: Cell
    radius_cells: int
    value: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RefinementResult:
    updated_data: NDArray[np.int8]
    changed_mask: NDArray[np.bool_]
    target: InspectionTarget


def cell_to_metric(cell: Cell, config: OccupancyGridConfig) -> tuple[float, float]:
    row, column = cell
    if not (0 <= row < config.height and 0 <= column < config.width):
        raise ValueError("cell is outside grid")
    return (config.x_min_m + (column + 0.5) * config.resolution_m,
            config.y_min_m + (row + 0.5) * config.resolution_m)


def metric_to_cell(x_m: float, y_m: float, config: OccupancyGridConfig) -> Cell:
    grid = OccupancyGrid(config)
    cell = grid.metric_to_cell(x_m, y_m)
    if cell is None:
        raise ValueError("metric position is outside grid")
    return cell


def prepare_planning_grid(grid: OccupancyGrid | NDArray[np.integer],
                          config: OccupancyGridConfig | None = None, *,
                          planner: PlannerConfig = PlannerConfig()) -> PreparedPlanningGrid:
    if isinstance(grid, OccupancyGrid):
        data = grid.data.copy()
        map_config = grid.config
    else:
        if config is None:
            raise ValueError("config is required when passing a raw grid array")
        data = np.asarray(grid, dtype=np.int8).copy()
        map_config = config
    if data.shape != (map_config.height, map_config.width):
        raise ValueError("occupancy array shape does not match grid configuration")
    if not np.isin(data, (UNKNOWN, FREE, OCCUPIED)).all():
        raise ValueError("occupancy grid contains unsupported values")
    obstacle = data == OCCUPIED
    if planner.unknown_policy == "block":
        obstacle |= data == UNKNOWN
    distance_pixels = cv2.distanceTransform((~obstacle).astype(np.uint8), cv2.DIST_L2, 5)
    clearance = distance_pixels * map_config.resolution_m
    # The local map boundary is unknown world, not free space: keep the entire
    # circular footprint inside it even when no explicit occupied border exists.
    rows, columns = np.indices(data.shape)
    boundary_clearance = np.minimum.reduce((
        rows + 0.5, data.shape[0] - rows - 0.5,
        columns + 0.5, data.shape[1] - columns - 0.5,
    )) * map_config.resolution_m
    clearance = np.minimum(clearance, boundary_clearance)
    inflated = clearance <= planner.robot_radius_m + 1e-6
    traversable = ~inflated
    return PreparedPlanningGrid(data, map_config, inflated, traversable, clearance)


def plan_candidates(prepared: PreparedPlanningGrid, start: Cell, goal: Cell, *,
                    planner: PlannerConfig = PlannerConfig(),
                    start_heading_rad: float | None = None) -> list[PathCandidate]:
    _validate_endpoint(start, "start", prepared)
    _validate_endpoint(goal, "goal", prepared)
    penalty = np.zeros(prepared.data.shape, dtype=np.float64)
    accepted: list[tuple[Cell, ...]] = []
    for _ in range(planner.candidate_attempts):
        path = _astar(prepared, start, goal, planner, penalty, start_heading_rad)
        if path is None:
            break
        if all(path_overlap(path, existing) <= planner.max_path_overlap for existing in accepted):
            accepted.append(path)
            if len(accepted) >= planner.candidate_count:
                break
        _add_corridor_penalty(penalty, path, prepared.config, planner)
    candidates = [PathCandidate(path, score_path(path, prepared, planner, start_heading_rad))
                  for path in accepted]
    candidates.sort(key=lambda candidate: candidate.score.total)
    return [PathCandidate(candidate.cells, candidate.score, rank + 1)
            for rank, candidate in enumerate(candidates)]


def plan_metric_candidates(prepared: PreparedPlanningGrid, start: PlanningPose,
                           goal: PlanningPose | tuple[float, float], *,
                           planner: PlannerConfig = PlannerConfig()) -> list[PathCandidate]:
    """Plan from vehicle-frame metric positions using the same grid planner."""
    goal_xy = (goal.x_m, goal.y_m) if isinstance(goal, PlanningPose) else goal
    start_cell = metric_to_cell(start.x_m, start.y_m, prepared.config)
    goal_cell = metric_to_cell(float(goal_xy[0]), float(goal_xy[1]), prepared.config)
    return plan_candidates(prepared, start_cell, goal_cell, planner=planner,
                           start_heading_rad=start.heading_rad)


def score_path(path: tuple[Cell, ...], prepared: PreparedPlanningGrid,
               planner: PlannerConfig = PlannerConfig(),
               start_heading_rad: float | None = None) -> PathScore:
    if not path:
        raise ValueError("path cannot be empty")
    resolution = prepared.config.resolution_m
    steps = np.array([np.hypot(b[0] - a[0], b[1] - a[1]) * resolution
                      for a, b in zip(path, path[1:])], dtype=float)
    length = float(np.sum(steps))
    cells = np.asarray(path, dtype=int)
    clearance = prepared.clearance_m[cells[:, 0], cells[:, 1]]
    segment_clearance = clearance[1:] if len(clearance) > 1 else clearance
    clearance_cost = float(np.sum(steps / np.maximum(segment_clearance, resolution))) if len(steps) else 0.0
    states = prepared.data[cells[:, 0], cells[:, 1]]
    segment_unknown = states[1:] == UNKNOWN if len(states) > 1 else states == UNKNOWN
    unknown_distance = float(np.sum(steps[segment_unknown])) if len(steps) else 0.0
    segment_narrow = segment_clearance < planner.narrow_clearance_m
    narrow_distance = float(np.sum(steps[segment_narrow])) if len(steps) else 0.0
    heading_change = _heading_change(path, start_heading_rad)
    weights = planner.weights
    unknown_cost = weights.unknown * unknown_distance if planner.unknown_policy == "penalize" else 0.0
    difficulty = (weights.clearance * clearance_cost + unknown_cost
                  + weights.narrow * narrow_distance + weights.heading_change * heading_change)
    total = weights.length * length + difficulty
    return PathScore(
        length, clearance_cost, float(np.min(clearance)), float(np.mean(clearance)),
        unknown_distance, unknown_distance / length if length else 0.0,
        narrow_distance, narrow_distance / length if length else 0.0,
        heading_change, difficulty, total,
    )


def path_overlap(first: tuple[Cell, ...], second: tuple[Cell, ...]) -> float:
    a, b = set(first), set(second)
    return float(len(a & b) / min(len(a), len(b))) if a and b else 0.0


def path_is_valid(path: tuple[Cell, ...], prepared: PreparedPlanningGrid) -> bool:
    if not path:
        return False
    previous: Cell | None = None
    for cell in path:
        row, column = cell
        if not (0 <= row < prepared.data.shape[0] and 0 <= column < prepared.data.shape[1]):
            return False
        if not prepared.traversable[cell]:
            return False
        if previous is not None and max(abs(row - previous[0]), abs(column - previous[1])) != 1:
            return False
        previous = cell
    return True


def select_inspection_targets(prepared: PreparedPlanningGrid,
                              candidates: list[PathCandidate], *,
                              planner: PlannerConfig = PlannerConfig(),
                              max_targets: int = 2) -> list[InspectionTarget]:
    if not candidates or max_targets < 1:
        return []
    scores = np.zeros(prepared.data.shape, dtype=np.float64)
    reasons: dict[Cell, set[str]] = {}
    radius = max(1, round(planner.inspection_radius_m / prepared.config.resolution_m))
    occurrence = np.zeros(prepared.data.shape, dtype=np.int16)
    for candidate in candidates[:min(3, len(candidates))]:
        for cell in set(candidate.cells):
            occurrence[cell] += 1
    top = candidates[0]
    for index, cell in enumerate(top.cells):
        if prepared.data[cell] == UNKNOWN:
            scores[cell] += 8.0
            reasons.setdefault(cell, set()).add("unknown_on_best_path")
        if prepared.clearance_m[cell] < planner.narrow_clearance_m:
            scores[cell] += 3.0
            reasons.setdefault(cell, set()).add("narrow_passage")
        if 0 < occurrence[cell] < min(3, len(candidates)):
            scores[cell] += 3.5
            reasons.setdefault(cell, set()).add("candidate_divergence")
        # Avoid spending an inspection immediately at start/goal.
        if index < 3 or index >= len(top.cells) - 2:
            scores[cell] *= 0.25

    # Unknown cells close enough to affect any leading route.
    path_mask = np.zeros(prepared.data.shape, np.uint8)
    for candidate in candidates[:min(3, len(candidates))]:
        for cell in candidate.cells:
            path_mask[cell] = 1
    near_paths = cv2.dilate(path_mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    nearby_unknown = near_paths & (prepared.data == UNKNOWN)
    scores[nearby_unknown] += 2.0
    for row, column in np.argwhere(nearby_unknown):
        reasons.setdefault((int(row), int(column)), set()).add("unknown_near_candidate")

    # Small isolated occupied returns next to a route are plausible ToF false positives.
    occupied = (prepared.data == OCCUPIED).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, 8)
    suspicious = np.zeros_like(occupied, dtype=bool)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] <= 3:
            suspicious |= labels == label
    suspicious_near = suspicious & near_paths
    scores[suspicious_near] += 7.0
    for row, column in np.argwhere(suspicious_near):
        reasons.setdefault((int(row), int(column)), set()).add("isolated_tof_return")

    targets: list[InspectionTarget] = []
    suppressed = np.zeros(prepared.data.shape, bool)
    for _ in range(max_targets):
        available = np.where(suppressed, -np.inf, scores)
        flat = int(np.argmax(available))
        value = float(available.flat[flat])
        if not np.isfinite(value) or value <= 0:
            break
        center = tuple(int(v) for v in np.unravel_index(flat, scores.shape))
        targets.append(InspectionTarget(center, radius, value,
                                        tuple(sorted(reasons.get(center, {"route_uncertainty"})))))
        rr, cc = np.ogrid[:scores.shape[0], :scores.shape[1]]
        suppressed |= (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= (2 * radius) ** 2
    return targets


def apply_simulated_refinement(rough_data: NDArray[np.integer],
                               truth_data: NDArray[np.integer],
                               target: InspectionTarget) -> RefinementResult:
    rough = np.asarray(rough_data, dtype=np.int8)
    truth = np.asarray(truth_data, dtype=np.int8)
    if rough.shape != truth.shape:
        raise ValueError("rough and truth maps must have equal shape")
    row, column = target.center
    if not (0 <= row < rough.shape[0] and 0 <= column < rough.shape[1]):
        raise ValueError("inspection target is outside map")
    rr, cc = np.ogrid[:rough.shape[0], :rough.shape[1]]
    region = (rr - row) ** 2 + (cc - column) ** 2 <= target.radius_cells ** 2
    updated = rough.copy()
    updated[region] = truth[region]
    changed = region & (updated != rough)
    return RefinementResult(updated, changed, target)


def _validate_endpoint(cell: Cell, label: str, prepared: PreparedPlanningGrid) -> None:
    row, column = cell
    if not (0 <= row < prepared.data.shape[0] and 0 <= column < prepared.data.shape[1]):
        raise ValueError(f"{label} is outside grid")
    if not prepared.traversable[cell]:
        raise ValueError(f"{label} is occupied, inflated, or disallowed unknown")


_MOVES: tuple[tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def _astar(prepared: PreparedPlanningGrid, start: Cell, goal: Cell,
           planner: PlannerConfig, penalty: NDArray[np.float64],
           start_heading_rad: float | None) -> tuple[Cell, ...] | None:
    # One best arrival per cell keeps this prototype fast enough for repeated
    # diverse searches. Incoming direction is retained for turn cost, although
    # this is not the globally exact orientation-expanded state formulation.
    shape = prepared.data.shape
    costs = np.full(shape, np.inf, dtype=np.float64)
    arrival_direction = np.full(shape, -1, dtype=np.int8)
    parent_row = np.full(shape, -1, dtype=np.int32)
    parent_column = np.full(shape, -1, dtype=np.int32)
    costs[start] = 0.0
    if start_heading_rad is not None:
        arrival_direction[start] = _heading_to_direction(start_heading_rad)
    queue: list[tuple[float, int, int, int]] = []
    sequence = count()
    heappush(queue, (_heuristic(start, goal, prepared.config.resolution_m, planner),
                     next(sequence), start[0], start[1]))
    found = False
    resolution = prepared.config.resolution_m
    while queue:
        priority, _, row, column = heappop(queue)
        expected = costs[row, column] + _heuristic((row, column), goal, resolution, planner)
        if priority > expected + 1e-10:
            continue
        if (row, column) == goal:
            found = True
            break
        previous_direction = int(arrival_direction[row, column])
        for direction, (dr, dc) in enumerate(_MOVES):
            neighbor = (row + dr, column + dc)
            if not (0 <= neighbor[0] < prepared.data.shape[0]
                    and 0 <= neighbor[1] < prepared.data.shape[1]):
                continue
            if not prepared.traversable[neighbor]:
                continue
            # Prevent diagonally cutting through obstacle corners.
            if dr and dc and (not prepared.traversable[row + dr, column]
                              or not prepared.traversable[row, column + dc]):
                continue
            step = np.hypot(dr, dc) * resolution
            clearance = max(prepared.clearance_m[neighbor], resolution)
            edge = planner.weights.length * step
            edge += planner.weights.clearance * step / clearance
            if prepared.data[neighbor] == UNKNOWN and planner.unknown_policy == "penalize":
                edge += planner.weights.unknown * step
            if clearance < planner.narrow_clearance_m:
                edge += planner.weights.narrow * step
            if previous_direction >= 0:
                edge += planner.weights.heading_change * _direction_change(previous_direction, direction)
            edge += penalty[neighbor] * step
            tentative = costs[row, column] + edge
            if tentative + 1e-12 < costs[neighbor]:
                costs[neighbor] = tentative
                arrival_direction[neighbor] = direction
                parent_row[neighbor] = row
                parent_column[neighbor] = column
                heappush(queue, (tentative + _heuristic(neighbor, goal, resolution, planner),
                                 next(sequence), neighbor[0], neighbor[1]))
    if not found:
        return None
    cells: list[Cell] = [goal]
    current = goal
    while current != start:
        current = (int(parent_row[current]), int(parent_column[current]))
        if current[0] < 0:
            return None
        cells.append(current)
    return tuple(reversed(cells))


def _heuristic(cell: Cell, goal: Cell, resolution: float, planner: PlannerConfig) -> float:
    return planner.weights.length * np.hypot(goal[0] - cell[0], goal[1] - cell[1]) * resolution


def _direction_change(first: int, second: int) -> float:
    difference = abs(first - second)
    return min(difference, 8 - difference) * (pi / 4)


def _heading_to_direction(heading_rad: float) -> int:
    if not np.isfinite(heading_rad):
        raise ValueError("start heading must be finite")
    headings = np.asarray([atan2(dc, dr) for dr, dc in _MOVES])
    differences = np.abs((headings - heading_rad + pi) % (2 * pi) - pi)
    return int(np.argmin(differences))


def _heading_change(path: tuple[Cell, ...], start_heading_rad: float | None = None) -> float:
    if len(path) < 3:
        return 0.0
    headings = [atan2(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    total = 0.0
    if start_heading_rad is not None:
        difference = (headings[0] - start_heading_rad + pi) % (2 * pi) - pi
        total += abs(difference)
    for first, second in zip(headings, headings[1:]):
        difference = (second - first + pi) % (2 * pi) - pi
        total += abs(difference)
    return total


def _add_corridor_penalty(penalty: NDArray[np.float64], path: tuple[Cell, ...],
                          config: OccupancyGridConfig, planner: PlannerConfig) -> None:
    mask = np.zeros(penalty.shape, np.uint8)
    for cell in path:
        mask[cell] = 1
    radius = max(0, round(planner.diversity_radius_m / config.resolution_m))
    if radius:
        mask = cv2.dilate(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    penalty[mask > 0] += planner.diversity_penalty
