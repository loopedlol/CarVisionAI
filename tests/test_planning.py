import numpy as np
import pytest

from carvision.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGridConfig
from carvision.planning import (
    InspectionTarget,
    PlannerConfig,
    PlanningPose,
    ScoreWeights,
    apply_simulated_refinement,
    cell_to_metric,
    metric_to_cell,
    path_is_valid,
    path_overlap,
    plan_candidates,
    plan_metric_candidates,
    prepare_planning_grid,
    score_path,
    select_inspection_targets,
)
from carvision.planning_mock import SCENARIO_NAMES, make_planning_scenario


def quick_planner(**changes: object) -> PlannerConfig:
    values = dict(robot_radius_m=0.15, candidate_count=3, candidate_attempts=7,
                  diversity_radius_m=0.25, diversity_penalty=8.0)
    values.update(changes)
    return PlannerConfig(**values)


def test_metric_cell_conversion_round_trip_and_boundaries() -> None:
    config = OccupancyGridConfig(-2, 2, -1, 3, 0.5)
    for cell in ((0, 0), (3, 4), (7, 7)):
        assert metric_to_cell(*cell_to_metric(cell, config), config) == cell
    with pytest.raises(ValueError):
        cell_to_metric((8, 0), config)
    with pytest.raises(ValueError):
        metric_to_cell(2.0, 0.0, config)
    data = np.full((8, 8), FREE, np.int8)
    planner = quick_planner(robot_radius_m=0, candidate_count=1, candidate_attempts=1)
    prepared = prepare_planning_grid(data, config, planner=planner)
    start_xy, goal_xy = cell_to_metric((1, 1), config), cell_to_metric((6, 6), config)
    paths = plan_metric_candidates(prepared, PlanningPose(*start_xy, heading_rad=0), goal_xy,
                                   planner=planner)
    assert paths[0].cells[0] == (1, 1) and paths[0].cells[-1] == (6, 6)


def test_robot_inflation_and_clearance() -> None:
    config = OccupancyGridConfig(0, 7, 0, 7, 1.0)
    data = np.full((7, 7), FREE, np.int8)
    data[3, 3] = OCCUPIED
    prepared = prepare_planning_grid(data, config, planner=quick_planner(robot_radius_m=1.0))
    assert prepared.inflated_unsafe[3, 3]
    assert prepared.inflated_unsafe[3, 2]
    assert prepared.inflated_unsafe[2, 3]
    assert not prepared.inflated_unsafe[2, 2]
    assert prepared.clearance_m[3, 3] == 0
    assert prepared.clearance_m[3, 2] == pytest.approx(1.0)
    assert prepared.clearance_m[0, 0] == pytest.approx(0.5)


def test_open_map_planning_is_valid() -> None:
    scenario = make_planning_scenario("open")
    planner = quick_planner()
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    assert candidates
    assert path_is_valid(candidates[0].cells, prepared)
    assert candidates[0].cells[0] == scenario.start
    assert candidates[0].cells[-1] == scenario.goal


def test_unknown_policy_can_block_or_penalize() -> None:
    config = OccupancyGridConfig(0, 6, 0, 6, 1)
    data = np.full((6, 6), FREE, np.int8)
    data[3, :] = UNKNOWN
    blocked_cfg = quick_planner(unknown_policy="block", robot_radius_m=0, candidate_count=1,
                                candidate_attempts=1)
    blocked = prepare_planning_grid(data, config, planner=blocked_cfg)
    assert plan_candidates(blocked, (1, 2), (5, 2), planner=blocked_cfg) == []
    allowed_cfg = quick_planner(unknown_policy="penalize", robot_radius_m=0,
                                candidate_count=1, candidate_attempts=1)
    allowed = prepare_planning_grid(data, config, planner=allowed_cfg)
    candidate = plan_candidates(allowed, (1, 2), (5, 2), planner=allowed_cfg)[0]
    assert candidate.score.unknown_distance_m > 0


def test_blocked_unknown_is_footprint_inflated() -> None:
    config = OccupancyGridConfig(0, 7, 0, 7, 1)
    data = np.full((7, 7), FREE, np.int8)
    data[3, 3] = UNKNOWN
    planner = quick_planner(unknown_policy="block", robot_radius_m=1.0)
    prepared = prepare_planning_grid(data, config, planner=planner)
    assert prepared.inflated_unsafe[3, 3]
    assert prepared.inflated_unsafe[3, 2]
    assert not prepared.inflated_unsafe[2, 2]


def test_left_right_candidates_are_meaningfully_diverse() -> None:
    scenario = make_planning_scenario("left_right")
    planner = quick_planner(candidate_count=4, candidate_attempts=10, max_path_overlap=0.75)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    assert len(candidates) >= 2
    assert path_overlap(candidates[0].cells, candidates[1].cells) <= 0.75
    first_columns = np.asarray(candidates[0].cells)[:, 1]
    second_columns = np.asarray(candidates[1].cells)[:, 1]
    assert (first_columns.min() < 31 and second_columns.max() >= 49) or (
        second_columns.min() < 31 and first_columns.max() >= 49)


def test_scoring_exposes_unknown_clearance_and_heading_components() -> None:
    config = OccupancyGridConfig(0, 6, 0, 6, 1)
    data = np.full((6, 6), FREE, np.int8)
    data[2, 2] = UNKNOWN
    data[4, 4] = OCCUPIED
    planner = quick_planner(robot_radius_m=0, weights=ScoreWeights(1, 1, 4, 2, 1))
    prepared = prepare_planning_grid(data, config, planner=planner)
    straight = ((1, 1), (2, 2), (3, 3))
    bent = ((1, 1), (1, 2), (2, 3), (3, 3))
    straight_score = score_path(straight, prepared, planner)
    bent_score = score_path(bent, prepared, planner)
    assert straight_score.unknown_distance_m > 0
    assert bent_score.heading_change_rad > straight_score.heading_change_rad
    assert straight_score.total == pytest.approx(
        straight_score.length_m + straight_score.traversal_difficulty)


def test_inspection_target_prioritizes_unknown_route_region() -> None:
    scenario = make_planning_scenario("unknown_shortcut")
    planner = quick_planner(
        weights=ScoreWeights(1, 0.2, 0.2, 0.5, 0.1), inspection_radius_m=0.6)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    targets = select_inspection_targets(prepared, candidates, planner=planner, max_targets=1)
    assert targets
    target = targets[0]
    assert scenario.rough_data[target.center] == UNKNOWN
    assert "unknown_on_best_path" in target.reasons


def test_inspection_target_can_prioritize_isolated_tof_return() -> None:
    scenario = make_planning_scenario("false_tof_obstacle")
    planner = quick_planner(inspection_radius_m=0.7)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    targets = select_inspection_targets(prepared, candidates, planner=planner, max_targets=1)
    assert targets and "isolated_tof_return" in targets[0].reasons


def test_narrow_opening_can_be_closed_by_larger_footprint() -> None:
    scenario = make_planning_scenario("narrow_opening")
    small_cfg = quick_planner(robot_radius_m=0.15, candidate_count=1, candidate_attempts=1)
    small = prepare_planning_grid(scenario.rough_data, scenario.config, planner=small_cfg)
    assert plan_candidates(small, scenario.start, scenario.goal, planner=small_cfg)
    large_cfg = quick_planner(robot_radius_m=0.5, candidate_count=1, candidate_attempts=1)
    large = prepare_planning_grid(scenario.rough_data, scenario.config, planner=large_cfg)
    assert plan_candidates(large, scenario.start, scenario.goal, planner=large_cfg) == []


def test_confirmed_obstacle_changes_selected_route_after_refinement() -> None:
    scenario = make_planning_scenario("active_inspection")
    planner = quick_planner(weights=ScoreWeights(1, 0.35, 0.4, 1, 0.2),
                            inspection_radius_m=0.8)
    rough = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    initial = plan_candidates(rough, scenario.start, scenario.goal, planner=planner)
    target = select_inspection_targets(rough, initial, planner=planner, max_targets=1)[0]
    update = apply_simulated_refinement(scenario.rough_data, scenario.truth_data, target)
    refined = prepare_planning_grid(update.updated_data, scenario.config, planner=planner)
    replanned = plan_candidates(refined, scenario.start, scenario.goal, planner=planner)
    assert np.count_nonzero(update.changed_mask) > 0
    assert replanned
    assert initial[0].cells != replanned[0].cells
    assert path_is_valid(replanned[0].cells, refined)


def test_clearing_false_tof_obstacle_enables_shorter_route() -> None:
    scenario = make_planning_scenario("stereo_clears_obstacle")
    planner = quick_planner(candidate_count=1, candidate_attempts=1)
    rough = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    before = plan_candidates(rough, scenario.start, scenario.goal, planner=planner)[0]
    target = InspectionTarget((49, 40), 5, 1.0, ("isolated_tof_return",))
    update = apply_simulated_refinement(scenario.rough_data, scenario.truth_data, target)
    refined = prepare_planning_grid(update.updated_data, scenario.config, planner=planner)
    after = plan_candidates(refined, scenario.start, scenario.goal, planner=planner)[0]
    assert after.score.length_m < before.score.length_m


def test_no_route_and_invalid_start_are_reported() -> None:
    no_route = make_planning_scenario("no_route")
    planner = quick_planner(candidate_count=1, candidate_attempts=1)
    prepared = prepare_planning_grid(no_route.rough_data, no_route.config, planner=planner)
    assert plan_candidates(prepared, no_route.start, no_route.goal, planner=planner) == []
    invalid = make_planning_scenario("invalid_start")
    prepared_invalid = prepare_planning_grid(invalid.rough_data, invalid.config, planner=planner)
    with pytest.raises(ValueError, match="start"):
        plan_candidates(prepared_invalid, invalid.start, invalid.goal, planner=planner)


def test_all_requested_mock_scenarios_exist() -> None:
    expected = {"open", "blocked_direct", "left_right", "unknown_shortcut", "narrow_opening",
                "false_tof_obstacle", "stereo_confirms_obstacle", "stereo_clears_obstacle",
                "no_route", "invalid_start"}
    assert expected <= set(SCENARIO_NAMES)
    for name in expected:
        scenario = make_planning_scenario(name)
        assert scenario.rough_data.shape == scenario.truth_data.shape
