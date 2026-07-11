"""Temporal occupancy stabilization and route-aware replanning triggers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter

import cv2
import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGridConfig
from .planning import (
    Cell, InspectionTarget, PathCandidate, PlannerConfig, path_is_valid,
    plan_candidates, prepare_planning_grid, score_path,
)


@dataclass(frozen=True)
class TemporalMapConfig:
    occupied_confirmations: int = 2
    free_confirmations: int = 3
    unknown_confirmations: int = 3
    minimum_observation_confidence: float = 0.45
    confidence_gain: float = 0.12
    confidence_decay: float = 0.22
    transition_confidence: float = 0.6
    confidence_change_threshold: float = 0.1

    def __post_init__(self) -> None:
        if min(self.occupied_confirmations, self.free_confirmations,
               self.unknown_confirmations) < 1:
            raise ValueError("confirmation counts must be positive")
        values = (self.minimum_observation_confidence, self.confidence_gain,
                  self.confidence_decay, self.transition_confidence,
                  self.confidence_change_threshold)
        if not all(0 <= value <= 1 for value in values):
            raise ValueError("confidence settings must lie in [0, 1]")


@dataclass
class TemporalMapState:
    stable: NDArray[np.int8]
    confidence: NDArray[np.float32]
    observation_count: NDArray[np.uint16]
    pending_state: NDArray[np.int8]
    pending_count: NDArray[np.uint16]
    last_observed: NDArray[np.int8]
    frame_index: int = 0

    @classmethod
    def initialize(cls, occupancy: NDArray[np.integer], *,
                   initial_confidence: float = 0.8) -> "TemporalMapState":
        data = _validated_occupancy(occupancy)
        if not 0 <= initial_confidence <= 1:
            raise ValueError("initial_confidence must lie in [0,1]")
        return cls(data.copy(), np.full(data.shape, initial_confidence, np.float32),
                   np.ones(data.shape, np.uint16), data.copy(),
                   np.zeros(data.shape, np.uint16), data.copy(), 0)

    def update(self, observed: NDArray[np.integer], *,
               config: TemporalMapConfig = TemporalMapConfig(),
               observation_confidence: float | NDArray[np.floating] = 1.0,
               evidence_weight: int = 1) -> "MapChangeSet":
        observation = _validated_occupancy(observed)
        if observation.shape != self.stable.shape or evidence_weight < 1:
            raise ValueError("observation shape must match and evidence_weight must be positive")
        confidence = np.broadcast_to(np.asarray(observation_confidence, dtype=np.float32),
                                     observation.shape)
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("observation confidence must be finite in [0,1]")
        previous_stable = self.stable.copy()
        previous_confidence = self.confidence.copy()
        raw_changed = observation != self.last_observed
        raw_newly_occupied = (observation == OCCUPIED) & (previous_stable != OCCUPIED)
        strong = confidence >= config.minimum_observation_confidence
        agrees = observation == self.stable
        self.confidence[agrees] = np.minimum(
            1.0, self.confidence[agrees] + config.confidence_gain * confidence[agrees])
        disagrees = ~agrees
        self.confidence[disagrees] = np.maximum(
            0.0, self.confidence[disagrees] - config.confidence_decay * confidence[disagrees])
        same_pending = disagrees & strong & (self.pending_state == observation)
        new_pending = disagrees & strong & ~same_pending
        self.pending_count[same_pending] = np.minimum(
            np.iinfo(np.uint16).max,
            self.pending_count[same_pending].astype(np.uint32) + evidence_weight).astype(np.uint16)
        self.pending_state[new_pending] = observation[new_pending]
        self.pending_count[new_pending] = np.uint16(min(evidence_weight, np.iinfo(np.uint16).max))
        self.pending_count[agrees] = 0
        self.pending_state[agrees] = self.stable[agrees]
        thresholds = np.full(observation.shape, config.unknown_confirmations, np.uint16)
        thresholds[observation == OCCUPIED] = config.occupied_confirmations
        thresholds[observation == FREE] = config.free_confirmations
        transition = disagrees & strong & (self.pending_count >= thresholds)
        self.stable[transition] = observation[transition]
        self.confidence[transition] = config.transition_confidence
        self.pending_count[transition] = 0
        self.pending_state[transition] = self.stable[transition]
        self.observation_count = np.minimum(
            np.iinfo(np.uint16).max,
            self.observation_count.astype(np.uint32) + 1).astype(np.uint16)
        self.last_observed = observation.copy()
        self.frame_index += 1
        stable_changed = self.stable != previous_stable
        # Report confidence changes only where the observation challenges the
        # stable state. Routine confidence saturation on every agreeing cell is
        # bookkeeping, not a navigation event.
        confidence_changed = ((np.abs(self.confidence - previous_confidence)
                               >= config.confidence_change_threshold)
                              & (observation != previous_stable) & ~stable_changed)
        return MapChangeSet(
            self.frame_index, observation.copy(), previous_stable, self.stable.copy(),
            raw_changed, raw_newly_occupied, stable_changed,
            stable_changed & (self.stable == OCCUPIED),
            stable_changed & (self.stable == FREE),
            stable_changed & (self.stable == UNKNOWN),
            confidence_changed, disagrees & ~transition,
        )


@dataclass(frozen=True)
class MapChangeSet:
    frame_index: int
    observed: NDArray[np.int8]
    previous_stable: NDArray[np.int8]
    stable: NDArray[np.int8]
    raw_changed: NDArray[np.bool_]
    raw_newly_occupied: NDArray[np.bool_]
    stable_changed: NDArray[np.bool_]
    newly_occupied: NDArray[np.bool_]
    newly_free: NDArray[np.bool_]
    newly_unknown: NDArray[np.bool_]
    confidence_changed: NDArray[np.bool_]
    suppressed_transient: NDArray[np.bool_]

    def counts(self) -> dict[str, int]:
        return {name: int(np.count_nonzero(getattr(self, name))) for name in (
            "raw_changed", "stable_changed", "newly_occupied", "newly_free",
            "newly_unknown", "confidence_changed", "suppressed_transient")}


@dataclass(frozen=True)
class RouteContext:
    shape: tuple[int, int]
    selected_path: tuple[Cell, ...]
    candidates: tuple[PathCandidate, ...]
    start: Cell
    goal: Cell
    path_mask: NDArray[np.bool_]
    footprint_corridor: NDArray[np.bool_]
    safety_corridor: NDArray[np.bool_]
    stopping_corridor: NDArray[np.bool_]
    candidate_mask: NDArray[np.bool_]
    divergence_mask: NDArray[np.bool_]
    inspection_mask: NDArray[np.bool_]
    goal_mask: NDArray[np.bool_]


@dataclass(frozen=True)
class RouteRelevanceConfig:
    safety_margin_m: float = 0.25
    stopping_distance_m: float = 1.5
    stopping_half_width_m: float = 0.35
    goal_radius_m: float = 0.5

    def __post_init__(self) -> None:
        if min(self.safety_margin_m, self.stopping_distance_m,
               self.stopping_half_width_m, self.goal_radius_m) < 0:
            raise ValueError("route relevance distances must be non-negative")


def build_route_context(shape: tuple[int, int], config: OccupancyGridConfig,
                        candidates: list[PathCandidate], start: Cell, goal: Cell, *,
                        planner: PlannerConfig,
                        inspection_targets: list[InspectionTarget] | None = None,
                        relevance: RouteRelevanceConfig = RouteRelevanceConfig()) -> RouteContext:
    if not candidates:
        raise ValueError("route context requires at least one candidate")
    if shape != (config.height, config.width):
        raise ValueError("route context shape does not match grid")
    selected = candidates[0].cells
    path = _cells_mask(shape, selected)
    footprint = _dilate_metric(path, planner.robot_radius_m, config.resolution_m)
    safety = _dilate_metric(path, planner.robot_radius_m + relevance.safety_margin_m,
                            config.resolution_m)
    safety_radius = int(np.ceil((planner.robot_radius_m + relevance.safety_margin_m)
                                / config.resolution_m))
    candidate_mask = np.zeros(shape, bool)
    occurrence = np.zeros(shape, np.uint16)
    for candidate in candidates:
        mask = _cells_mask(shape, candidate.cells)
        candidate_mask |= mask
        occurrence += mask.astype(np.uint16)
    divergence = (occurrence > 0) & (occurrence < len(candidates))
    divergence = _dilate(divergence, max(1, safety_radius // 2))
    stopping_cells: list[Cell] = [selected[0]]
    distance = 0.0
    for first, second in zip(selected, selected[1:]):
        distance += np.hypot(second[0] - first[0], second[1] - first[1]) * config.resolution_m
        if distance > relevance.stopping_distance_m:
            break
        stopping_cells.append(second)
    stopping = _dilate_metric(_cells_mask(shape, stopping_cells),
                              relevance.stopping_half_width_m, config.resolution_m)
    inspection = np.zeros(shape, bool)
    for target in inspection_targets or []:
        inspection |= _circle_mask(shape, target.center, target.radius_cells)
    goal_mask = _circle_mask(shape, goal,
                             int(np.ceil(relevance.goal_radius_m / config.resolution_m)))
    return RouteContext(shape, selected, tuple(candidates), start, goal, path,
                        footprint, safety, stopping, candidate_mask, divergence,
                        inspection, goal_mask)


class PlanningAction(str, Enum):
    NONE = "no_action"
    RESCORE = "update_route_metrics"
    REEVALUATE = "reevaluate_candidates"
    FAST_REPLAN = "fast_single_path_replan"
    FULL_REPLAN = "full_multi_candidate_replan"
    STOP = "immediate_stop_invalid_route"


@dataclass(frozen=True)
class TriggerConfig:
    corruption_fraction: float = 0.35
    rescore_importance: float = 1.5
    reevaluate_importance: float = 5.0
    full_replan_importance: float = 16.0
    immediate_raw_stop: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.corruption_fraction <= 1:
            raise ValueError("corruption_fraction must lie in (0,1]")
        if not (0 <= self.rescore_importance <= self.reevaluate_importance
                <= self.full_replan_importance):
            raise ValueError("importance thresholds must be ordered")


@dataclass(frozen=True)
class TriggerDecision:
    action: PlanningAction
    reasons: tuple[str, ...]
    route_valid: bool
    valid_candidate_count: int
    importance: float
    route_relevant_changed_cells: int
    stopping_changed_cells: int
    counts: dict[str, int]


def analyze_map_change(change: MapChangeSet, context: RouteContext,
                       planner: PlannerConfig, map_config: OccupancyGridConfig, *,
                       trigger: TriggerConfig = TriggerConfig()) -> TriggerDecision:
    if change.stable.shape != context.shape:
        raise ValueError("change set and route context shape mismatch")
    counts = change.counts()
    raw_fraction = counts["raw_changed"] / change.stable.size
    raw_stop = change.raw_newly_occupied & context.stopping_corridor
    stable_stop = change.newly_occupied & context.stopping_corridor
    route_valid = fast_validate_path(change.stable, context.footprint_corridor, planner)
    valid_candidates = sum(fast_validate_candidate(change.stable, candidate, map_config, planner)
                           for candidate in context.candidates)
    importance_map = _importance_map(context)
    severity = np.zeros(context.shape, np.float64)
    severity[change.confidence_changed] = 0.15
    severity[change.newly_unknown] = 0.4
    severity[change.newly_free] = 0.65
    severity[change.newly_occupied] = 1.0
    importance = float(np.sum(importance_map * severity))
    relevant = change.stable_changed & (context.safety_corridor | context.candidate_mask
                                        | context.divergence_mask | context.inspection_mask
                                        | context.goal_mask)
    reasons: list[str] = []
    action = PlanningAction.NONE
    if raw_fraction >= trigger.corruption_fraction:
        action = PlanningAction.STOP
        reasons.append(f"widespread raw map corruption ({raw_fraction:.1%})")
    elif trigger.immediate_raw_stop and np.any(raw_stop):
        action = PlanningAction.STOP
        reasons.append(f"raw occupied evidence in stopping corridor ({np.count_nonzero(raw_stop)} cells)")
    elif np.any(stable_stop):
        action = PlanningAction.STOP
        reasons.append("persistent occupied transition in stopping corridor")
    elif not route_valid:
        if valid_candidates > 0:
            action = PlanningAction.REEVALUATE
            reasons.append(f"selected path invalid; {valid_candidates} existing candidates remain valid")
        else:
            action = PlanningAction.FAST_REPLAN
            reasons.append("selected path invalid and no existing candidate remains valid")
    elif np.any(change.newly_free & (context.divergence_mask | context.inspection_mask
                                     | context.safety_corridor)):
        action = PlanningAction.FULL_REPLAN
        reasons.append("persistent newly free region may create a better route")
    elif importance >= trigger.full_replan_importance:
        action = PlanningAction.FULL_REPLAN
        reasons.append(f"route-weighted change importance {importance:.1f} exceeds full threshold")
    elif np.any(change.stable_changed & context.candidate_mask) or importance >= trigger.reevaluate_importance:
        action = PlanningAction.REEVALUATE
        reasons.append("persistent change affects one or more candidate routes")
    elif np.any(change.stable_changed & context.safety_corridor) or importance >= trigger.rescore_importance:
        action = PlanningAction.RESCORE
        if np.any(change.stable_changed & context.safety_corridor):
            reasons.append("stable change affects route cost or clearance")
        else:
            reasons.append("route-weighted confidence change warrants metric refresh")
    elif np.any(change.confidence_changed & (context.path_mask | context.inspection_mask)):
        action = PlanningAction.RESCORE
        reasons.append("confidence changed on route-relevant cells")
    elif counts["suppressed_transient"]:
        reasons.append(f"suppressed {counts['suppressed_transient']} nonpersistent cell observations")
    elif counts["stable_changed"]:
        reasons.append("stable changes are distant from all navigation-relevant regions")
    else:
        reasons.append("no stable navigation-relevant map change")
    if action == PlanningAction.STOP:
        route_valid = False
    return TriggerDecision(
        action, tuple(reasons), route_valid, valid_candidates, importance,
        int(np.count_nonzero(relevant)), int(np.count_nonzero(
            change.stable_changed & context.stopping_corridor)), counts)


def fast_validate_path(stable: NDArray[np.integer], footprint_corridor: NDArray[np.bool_],
                       planner: PlannerConfig) -> bool:
    data = np.asarray(stable)
    blocked = data == OCCUPIED
    if planner.unknown_policy == "block":
        blocked |= data == UNKNOWN
    return not np.any(blocked & footprint_corridor)


def fast_validate_candidate(stable: NDArray[np.integer], candidate: PathCandidate,
                            config: OccupancyGridConfig, planner: PlannerConfig) -> bool:
    path = _cells_mask(np.asarray(stable).shape, candidate.cells)
    footprint = _dilate_metric(path, planner.robot_radius_m, config.resolution_m)
    return fast_validate_path(stable, footprint, planner)


@dataclass(frozen=True)
class PlanningExecution:
    requested_action: PlanningAction
    executed_action: PlanningAction
    candidates: tuple[PathCandidate, ...]
    planning_time_ms: float
    route_valid: bool


def execute_planning_action(decision: TriggerDecision, stable: NDArray[np.integer],
                            context: RouteContext, config: OccupancyGridConfig,
                            planner: PlannerConfig) -> PlanningExecution:
    start_time = perf_counter()
    if decision.action == PlanningAction.STOP:
        return PlanningExecution(decision.action, decision.action, context.candidates, 0.0, False)
    prepared = None
    candidates = list(context.candidates)
    executed = decision.action
    if decision.action == PlanningAction.NONE:
        pass
    elif decision.action == PlanningAction.RESCORE:
        prepared = prepare_planning_grid(stable, config, planner=planner)
        candidates = [PathCandidate(candidate.cells, score_path(candidate.cells, prepared, planner))
                      for candidate in candidates]
        candidates.sort(key=lambda candidate: candidate.score.total)
    elif decision.action == PlanningAction.REEVALUATE:
        prepared = prepare_planning_grid(stable, config, planner=planner)
        candidates = [PathCandidate(candidate.cells, score_path(candidate.cells, prepared, planner))
                      for candidate in candidates if path_is_valid(candidate.cells, prepared)]
        candidates.sort(key=lambda candidate: candidate.score.total)
        if not candidates:
            executed = PlanningAction.FAST_REPLAN
    if executed == PlanningAction.FAST_REPLAN:
        prepared = prepared or prepare_planning_grid(stable, config, planner=planner)
        fast_config = replace(planner, candidate_count=1, candidate_attempts=1)
        candidates = plan_candidates(prepared, context.start, context.goal, planner=fast_config)
    elif executed == PlanningAction.FULL_REPLAN:
        prepared = prepared or prepare_planning_grid(stable, config, planner=planner)
        candidates = plan_candidates(prepared, context.start, context.goal, planner=planner)
    ranked = tuple(PathCandidate(candidate.cells, candidate.score, index + 1)
                   for index, candidate in enumerate(candidates))
    elapsed = (perf_counter() - start_time) * 1000
    return PlanningExecution(decision.action, executed, ranked, elapsed, bool(ranked))


def _importance_map(context: RouteContext) -> NDArray[np.float64]:
    # Distant cells deliberately carry almost no aggregate weight: even a large
    # far-away object should not outweigh one persistent cell on the route.
    importance = np.full(context.shape, 0.001, np.float64)
    importance[context.goal_mask] = np.maximum(importance[context.goal_mask], 4.0)
    importance[context.inspection_mask] = np.maximum(importance[context.inspection_mask], 3.5)
    importance[context.divergence_mask] = np.maximum(importance[context.divergence_mask], 3.0)
    importance[context.candidate_mask] = np.maximum(importance[context.candidate_mask], 2.0)
    importance[context.safety_corridor] = np.maximum(importance[context.safety_corridor], 5.0)
    importance[context.path_mask] = np.maximum(importance[context.path_mask], 10.0)
    importance[context.stopping_corridor] = np.maximum(importance[context.stopping_corridor], 25.0)
    return importance


def _validated_occupancy(data: NDArray[np.integer]) -> NDArray[np.int8]:
    value = np.asarray(data, dtype=np.int8)
    if value.ndim != 2 or not np.isin(value, (UNKNOWN, FREE, OCCUPIED)).all():
        raise ValueError("occupancy must be a 2D ternary grid")
    return value


def _cells_mask(shape: tuple[int, int], cells: tuple[Cell, ...] | list[Cell]) -> NDArray[np.bool_]:
    mask = np.zeros(shape, bool)
    for cell in cells:
        if not (0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]):
            raise ValueError("path cell outside map")
        mask[cell] = True
    return mask


def _dilate(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    if radius <= 0:
        return mask.copy()
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = ((x * x + y * y) <= radius * radius).astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _dilate_metric(mask: NDArray[np.bool_], radius_m: float,
                   resolution_m: float) -> NDArray[np.bool_]:
    radius_cells = int(np.ceil(radius_m / resolution_m))
    if radius_cells <= 0:
        return mask.copy()
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    kernel = ((x * x + y * y) * resolution_m ** 2 <= radius_m ** 2 + 1e-9).astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def _circle_mask(shape: tuple[int, int], center: Cell, radius: int) -> NDArray[np.bool_]:
    rows, columns = np.ogrid[:shape[0], :shape[1]]
    return (rows - center[0]) ** 2 + (columns - center[1]) ** 2 <= radius ** 2
