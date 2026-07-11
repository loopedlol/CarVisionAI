import numpy as np
import pytest

from carvision.map_events import PlanningAction
from carvision.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGridConfig
from carvision.planning import PathCandidate, PlannerConfig, ScoreWeights, plan_candidates, prepare_planning_grid
from carvision.trajectory import (
    TrajectoryAction, TrajectoryConfig, collision_indices,
    combine_event_and_trajectory_action, generate_trajectory, grid_path_to_metric,
    remove_duplicate_and_collinear, sample_polyline, segment_is_collision_free,
    simplify_line_of_sight, smooth_corners, stopping_distance_m, validate_trajectory,
)
from carvision.trajectory_mock import TRAJECTORY_SCENARIOS, make_trajectory_scenario


def configs() -> tuple[PlannerConfig, TrajectoryConfig]:
    return (PlannerConfig(robot_radius_m=0.15, candidate_count=1, candidate_attempts=1),
            TrajectoryConfig(robot_radius_m=0.15, safety_margin_m=0.02,
                             sample_spacing_m=0.03, corner_radius_m=0.40))


def prepared_scenario(name: str):
    scenario = make_trajectory_scenario(name)
    planner, trajectory = configs()
    prepared = prepare_planning_grid(scenario.data, scenario.config, planner=planner)
    return scenario, planner, trajectory, prepared


def test_grid_metric_conversion() -> None:
    scenario, _, _, _ = prepared_scenario("straight")
    metric = grid_path_to_metric(((0, 0), (1, 1)), scenario.config)
    np.testing.assert_allclose(metric[0], [-3.95, 0.05])
    np.testing.assert_allclose(metric[1], [-3.85, 0.15])


def test_duplicate_and_collinear_removal() -> None:
    points = np.array([[0, 0], [0, 0], [1, 1], [2, 2], [2, 3]], float)
    result = remove_duplicate_and_collinear(points)
    np.testing.assert_allclose(result, [[0, 0], [2, 2], [2, 3]])


def test_line_of_sight_simplifies_grid_diagonal() -> None:
    scenario, _, config, prepared = prepared_scenario("diagonal_zigzag")
    raw = grid_path_to_metric(scenario.path, scenario.config)
    simplified = simplify_line_of_sight(raw, prepared, config)
    assert len(raw) == 50
    assert len(simplified) == 2


def test_swept_footprint_collision_checks_segment_not_only_endpoints() -> None:
    config = OccupancyGridConfig(0, 5, 0, 5, 0.1)
    data = np.full((50, 50), FREE, np.int8); data[25, 25] = OCCUPIED
    planner, trajectory = configs()
    prepared = prepare_planning_grid(data, config, planner=planner)
    assert not segment_is_collision_free(np.array([1.0, 2.55]), np.array([4.0, 2.55]),
                                         prepared, trajectory)
    assert segment_is_collision_free(np.array([1.0, 1.0]), np.array([4.0, 1.0]),
                                     prepared, trajectory)


def test_corner_smoothing_rejects_unsafe_inside_cut() -> None:
    scenario, _, config, prepared = prepared_scenario("inside_corner_obstacle")
    raw = remove_duplicate_and_collinear(grid_path_to_metric(scenario.path, scenario.config))
    smoothed, rejected = smooth_corners(raw, prepared, config)
    assert rejected
    assert not collision_indices(sample_polyline(smoothed, config.sample_spacing_m), prepared, config)


def test_heading_curvature_and_angular_velocity_limits() -> None:
    scenario, _, config, prepared = prepared_scenario("corner_90")
    result = generate_trajectory(scenario.path, prepared, config)
    headings = np.unwrap([sample.heading_rad for sample in result.samples])
    assert abs(headings[-1] - headings[0]) == pytest.approx(np.pi / 2, abs=0.15)
    assert max(abs(sample.curvature_per_m) for sample in result.samples) > 0
    assert max(abs(sample.angular_velocity_rps) for sample in result.samples) <= config.max_angular_velocity_rps + 1e-9


def test_forward_acceleration_backward_braking_and_terminal_stop() -> None:
    scenario, _, config, prepared = prepared_scenario("straight")
    result = generate_trajectory(scenario.path, prepared, config)
    samples = result.samples
    assert samples[-1].linear_velocity_mps == 0
    assert samples[-1].angular_velocity_rps == 0
    expected_goal = grid_path_to_metric((scenario.path[-1],), scenario.config)[0]
    np.testing.assert_allclose([samples[-1].x_m, samples[-1].y_m], expected_goal)
    assert all(b.time_s > a.time_s for a, b in zip(samples, samples[1:]))
    for first, second in zip(samples, samples[1:]):
        distance = second.distance_m - first.distance_m
        assert second.linear_velocity_mps ** 2 <= first.linear_velocity_mps ** 2 + 2 * config.max_linear_acceleration_mps2 * distance + 1e-8
        assert first.linear_velocity_mps ** 2 <= second.linear_velocity_mps ** 2 + 2 * config.max_linear_deceleration_mps2 * distance + 1e-8


def test_angular_acceleration_and_deceleration_limits() -> None:
    scenario, _, config, prepared = prepared_scenario("corner_90")
    samples = generate_trajectory(scenario.path, prepared, config).samples
    for first, second in zip(samples, samples[1:]):
        dt = second.time_s - first.time_s
        limit = (config.max_angular_acceleration_rps2
                 if abs(second.angular_velocity_rps) >= abs(first.angular_velocity_rps)
                 else config.max_angular_deceleration_rps2)
        assert abs(second.angular_velocity_rps - first.angular_velocity_rps) <= limit * dt + 1e-8


def test_stopping_distance_includes_latency_braking_and_margin() -> None:
    _, config = configs()
    expected = 0.8 * config.reaction_latency_s + 0.8 ** 2 / (2 * config.max_linear_deceleration_mps2) + config.braking_safety_margin_m
    assert stopping_distance_m(0.8, config) == pytest.approx(expected)
    with pytest.raises(ValueError):
        stopping_distance_m(-0.1, config)


def test_low_clearance_and_unknown_reduce_speed() -> None:
    open_scenario, _, config, open_grid = prepared_scenario("straight")
    narrow_scenario, _, _, narrow_grid = prepared_scenario("narrow_corridor")
    unknown_scenario, _, _, unknown_grid = prepared_scenario("unknown_near_route")
    open_result = generate_trajectory(open_scenario.path, open_grid, config)
    narrow_result = generate_trajectory(narrow_scenario.path, narrow_grid, config)
    unknown_result = generate_trajectory(unknown_scenario.path, unknown_grid, config)
    open_peak = max(sample.linear_velocity_mps for sample in open_result.samples)
    assert max(sample.linear_velocity_mps for sample in narrow_result.samples) < open_peak
    unknown_speeds = [sample.linear_velocity_mps for sample in unknown_result.samples if sample.unknown]
    assert unknown_speeds and max(unknown_speeds) <= config.max_linear_velocity_mps * config.unknown_speed_factor + 1e-9


def test_generation_and_timestamps_are_deterministic() -> None:
    scenario, _, config, prepared = prepared_scenario("tight_s")
    first = generate_trajectory(scenario.path, prepared, config)
    second = generate_trajectory(scenario.path, prepared, config)
    np.testing.assert_allclose([sample.time_s for sample in first.samples],
                               [sample.time_s for sample in second.samples])
    np.testing.assert_allclose(first.smoothed_path, second.smoothed_path)


def test_unsafe_smoothing_falls_back_to_simplified_path() -> None:
    scenario, _, config, prepared = prepared_scenario("unsafe_smoothing")
    result = generate_trajectory(scenario.path, prepared, config)
    assert result.used_fallback
    assert result.rejected_smoothing_corners
    assert not collision_indices(np.asarray([(sample.x_m, sample.y_m)
                                              for sample in result.samples]), prepared, config)


def test_required_final_heading_adds_in_place_rotation_and_zero_terminal_command() -> None:
    scenario, _, config, prepared = prepared_scenario("final_heading")
    result = generate_trajectory(scenario.path, prepared, config,
                                 final_heading_rad=scenario.final_heading_rad)
    assert result.samples[-1].heading_rad == pytest.approx(np.pi / 2)
    assert result.samples[-1].linear_velocity_mps == 0
    assert result.samples[-1].angular_velocity_rps == 0
    assert any(sample.angular_velocity_rps != 0 for sample in result.samples[-3:-1])
    assert result.samples[-1].distance_m == result.samples[-2].distance_m
    rotation = [sample for sample in result.samples if sample.distance_m == result.samples[-1].distance_m]
    for first, second in zip(rotation, rotation[1:]):
        dt = second.time_s - first.time_s
        limit = (config.max_angular_acceleration_rps2
                 if abs(second.angular_velocity_rps) >= abs(first.angular_velocity_rps)
                 else config.max_angular_deceleration_rps2)
        assert abs(second.angular_velocity_rps - first.angular_velocity_rps) <= limit * dt + 1e-8


def test_updated_map_collision_inside_stopping_corridor_stops() -> None:
    scenario, planner, config, prepared = prepared_scenario("straight")
    trajectory = generate_trajectory(scenario.path, prepared, config)
    updated = scenario.data.copy(); updated[13:16, 39:42] = OCCUPIED
    validation = validate_trajectory(trajectory, updated, scenario.config, planner, config,
                                     current_index=0, current_speed_mps=0.8)
    assert validation.action == TrajectoryAction.STOP
    assert validation.stopping_corridor_collision_indices


def test_obstacle_beyond_braking_corridor_requires_replan_not_stop() -> None:
    scenario, planner, config, prepared = prepared_scenario("straight")
    trajectory = generate_trajectory(scenario.path, prepared, config)
    updated = scenario.data.copy(); updated[60:63, 39:42] = OCCUPIED
    validation = validate_trajectory(trajectory, updated, scenario.config, planner, config,
                                     current_index=0, current_speed_mps=0.2)
    assert validation.action == TrajectoryAction.REPLAN
    assert not validation.stopping_corridor_collision_indices


def test_map_can_require_trajectory_regeneration_while_raw_grid_path_remains_valid() -> None:
    scenario = make_trajectory_scenario("corner_90")
    planner = PlannerConfig(robot_radius_m=0.02, candidate_count=1, candidate_attempts=1)
    config = TrajectoryConfig(robot_radius_m=0.02, safety_margin_m=0.0,
                              sample_spacing_m=0.02, corner_radius_m=0.4)
    prepared = prepare_planning_grid(scenario.data, scenario.config, planner=planner)
    trajectory = generate_trajectory(scenario.path, prepared, config)
    raw_cells = set(scenario.path)
    updated = scenario.data.copy()
    chosen = None
    for sample in trajectory.samples:
        cell = (int((sample.y_m - scenario.config.y_min_m) / scenario.config.resolution_m),
                int((sample.x_m - scenario.config.x_min_m) / scenario.config.resolution_m))
        if cell not in raw_cells and 2 < cell[0] < 77 and 2 < cell[1] < 77:
            chosen = cell; break
    assert chosen is not None
    updated[chosen] = OCCUPIED
    validation = validate_trajectory(trajectory, updated, scenario.config, planner, config,
                                     current_index=0, current_speed_mps=0.05)
    assert validation.action == TrajectoryAction.REGENERATE
    assert validation.grid_path_valid


def test_event_and_trajectory_action_precedence() -> None:
    scenario, planner, config, prepared = prepared_scenario("straight")
    trajectory = generate_trajectory(scenario.path, prepared, config)
    valid = validate_trajectory(trajectory, scenario.data, scenario.config, planner, config)
    assert combine_event_and_trajectory_action(PlanningAction.NONE, valid) == TrajectoryAction.VALID
    assert combine_event_and_trajectory_action(PlanningAction.FULL_REPLAN, valid) == TrajectoryAction.REPLAN
    assert combine_event_and_trajectory_action(PlanningAction.STOP, valid) == TrajectoryAction.STOP


def test_all_requested_scenarios_exist() -> None:
    expected = {"straight", "corner_90", "diagonal_zigzag", "tight_s", "narrow_corridor",
                "inside_corner_obstacle", "unknown_near_route", "clearance_transition",
                "braking_obstacle", "beyond_braking", "trajectory_only_regeneration",
                "immediate_stop", "unsafe_smoothing", "fallback", "final_heading"}
    assert expected == set(TRAJECTORY_SCENARIOS)
