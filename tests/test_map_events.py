import numpy as np
import pytest

from carvision.map_events import (
    PlanningAction, RouteRelevanceConfig, TemporalMapConfig, TemporalMapState,
    analyze_map_change, build_route_context, fast_validate_path,
)
from carvision.mapping import FREE, OCCUPIED, UNKNOWN
from carvision.planning import PlannerConfig, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.map_event_mock import EVENT_SCENARIO_NAMES, make_event_scenario


@pytest.fixture(scope="module")
def navigation() -> tuple[object, PlannerConfig, object, list, object]:
    scenario = make_planning_scenario("open")
    planner = PlannerConfig(robot_radius_m=0.15, candidate_count=3, candidate_attempts=6)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    context = build_route_context(scenario.rough_data.shape, scenario.config, candidates,
                                  scenario.start, scenario.goal, planner=planner)
    return scenario, planner, prepared, candidates, context


def test_temporal_persistence_and_changed_classification() -> None:
    initial = np.full((4, 5), FREE, np.int8)
    state = TemporalMapState.initialize(initial)
    observed = initial.copy(); observed[2, 3] = OCCUPIED
    first = state.update(observed)
    assert not first.stable_changed.any()
    assert first.suppressed_transient[2, 3]
    second = state.update(observed)
    assert second.newly_occupied[2, 3]
    assert state.stable[2, 3] == OCCUPIED
    assert state.observation_count[2, 3] == 3


def test_hysteresis_requires_more_evidence_to_clear_than_occupy() -> None:
    initial = np.full((3, 3), FREE, np.int8); initial[1, 1] = OCCUPIED
    state = TemporalMapState.initialize(initial)
    clear = initial.copy(); clear[1, 1] = FREE
    assert not state.update(clear).stable_changed.any()
    assert not state.update(clear).stable_changed.any()
    third = state.update(clear)
    assert third.newly_free[1, 1]
    unknown = clear.copy(); unknown[0, 0] = UNKNOWN
    state.update(unknown); state.update(unknown)
    change = state.update(unknown)
    assert change.newly_unknown[0, 0]


def test_low_confidence_and_alternating_flicker_are_suppressed() -> None:
    initial = np.full((3, 3), FREE, np.int8)
    state = TemporalMapState.initialize(initial)
    occupied = initial.copy(); occupied[1, 1] = OCCUPIED
    for _ in range(4):
        change = state.update(occupied, observation_confidence=0.2)
    assert state.stable[1, 1] == FREE and change.suppressed_transient[1, 1]
    for index in range(8):
        state.update(occupied if index % 2 == 0 else initial)
    assert state.stable[1, 1] == FREE


def test_strong_stereo_evidence_uses_same_transition_machine() -> None:
    initial = np.full((3, 3), UNKNOWN, np.int8)
    state = TemporalMapState.initialize(initial)
    observed = initial.copy(); observed[1, 1] = OCCUPIED
    change = state.update(observed, evidence_weight=3)
    assert change.newly_occupied[1, 1]
    assert state.confidence[1, 1] == pytest.approx(0.6)


def test_route_and_stopping_corridor_masks(navigation: tuple) -> None:
    scenario, planner, _, candidates, context = navigation
    assert all(context.path_mask[cell] for cell in candidates[0].cells)
    assert context.stopping_corridor[context.start]
    assert context.stopping_corridor[candidates[0].cells[8]]
    assert not context.stopping_corridor[candidates[0].cells[-10]]
    assert np.count_nonzero(context.safety_corridor) > np.count_nonzero(context.path_mask)
    assert np.count_nonzero(context.divergence_mask) > 0


def test_one_frame_far_noise_does_not_trigger_replanning(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    irrelevant = ~(context.safety_corridor | context.candidate_mask | context.divergence_mask
                   | context.inspection_mask | context.goal_mask | context.stopping_corridor)
    cell = tuple(int(v) for v in np.argwhere(irrelevant & (scenario.rough_data == FREE))[0])
    observed = scenario.rough_data.copy(); observed[cell] = OCCUPIED
    decision = analyze_map_change(state.update(observed), context, planner, scenario.config)
    assert decision.action == PlanningAction.NONE
    assert "suppressed" in decision.reasons[0]


def test_small_route_change_outweighs_large_distant_change(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    route_cell = context.selected_path[len(context.selected_path) // 2]
    route_state = TemporalMapState.initialize(scenario.rough_data)
    route_observed = scenario.rough_data.copy(); route_observed[route_cell] = OCCUPIED
    route_state.update(route_observed)
    route_decision = analyze_map_change(route_state.update(route_observed), context,
                                        planner, scenario.config)

    far_state = TemporalMapState.initialize(scenario.rough_data)
    irrelevant = ~(context.safety_corridor | context.candidate_mask | context.divergence_mask
                   | context.inspection_mask | context.goal_mask | context.stopping_corridor)
    far_observed = scenario.rough_data.copy()
    far_cells = np.argwhere(irrelevant & (scenario.rough_data == FREE))[:200]
    far_observed[far_cells[:, 0], far_cells[:, 1]] = OCCUPIED
    far_state.update(far_observed)
    far_decision = analyze_map_change(far_state.update(far_observed), context,
                                      planner, scenario.config)
    assert route_decision.importance > far_decision.importance
    assert far_decision.action == PlanningAction.NONE
    assert route_decision.action in (PlanningAction.REEVALUATE, PlanningAction.FAST_REPLAN)


def test_raw_stopping_obstacle_causes_immediate_stop(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    cell = context.selected_path[6]
    observed = scenario.rough_data.copy(); observed[cell] = OCCUPIED
    decision = analyze_map_change(state.update(observed), context, planner, scenario.config)
    assert decision.action == PlanningAction.STOP
    assert not decision.route_valid and "raw occupied" in decision.reasons[0]


def test_first_nonstopping_path_observation_only_rescores(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    cell = context.selected_path[len(context.selected_path) // 2]
    observed = scenario.rough_data.copy(); observed[cell] = OCCUPIED
    decision = analyze_map_change(state.update(observed), context, planner, scenario.config)
    assert decision.action == PlanningAction.RESCORE
    assert decision.route_valid


def test_persistent_path_obstacle_invalidates_selected_route(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    cell = context.selected_path[len(context.selected_path) // 2]
    observed = scenario.rough_data.copy(); observed[cell] = OCCUPIED
    state.update(observed)
    change = state.update(observed)
    decision = analyze_map_change(change, context, planner, scenario.config)
    assert not decision.route_valid
    assert decision.action in (PlanningAction.REEVALUATE, PlanningAction.FAST_REPLAN)
    assert not fast_validate_path(change.stable, context.footprint_corridor, planner)


def test_persistent_newly_free_shortcut_requests_full_replan(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    cell = tuple(int(v) for v in np.argwhere(context.divergence_mask & (state.stable == FREE))[0])
    # First make the cell stably unknown, then clear it with directed evidence.
    unknown = state.stable.copy(); unknown[cell] = UNKNOWN
    state.update(unknown, evidence_weight=3)
    cleared = state.stable.copy(); cleared[cell] = FREE
    change = state.update(cleared, evidence_weight=3)
    decision = analyze_map_change(change, context, planner, scenario.config)
    assert change.newly_free[cell]
    assert decision.action == PlanningAction.FULL_REPLAN


def test_widespread_corruption_stops_deterministically(navigation: tuple) -> None:
    scenario, planner, _, _, context = navigation
    decisions = []
    for _ in range(2):
        state = TemporalMapState.initialize(scenario.rough_data)
        corrupted = scenario.rough_data.copy(); corrupted[5:60, 5:70] = UNKNOWN
        change = state.update(corrupted)
        decisions.append(analyze_map_change(change, context, planner, scenario.config))
    assert decisions[0].action == PlanningAction.STOP
    assert decisions[0].action == decisions[1].action
    assert decisions[0].reasons == decisions[1].reasons


def test_all_candidate_blockage_requests_fast_replan(navigation: tuple) -> None:
    scenario, planner, _, candidates, context = navigation
    state = TemporalMapState.initialize(scenario.rough_data)
    observed = scenario.rough_data.copy()
    for candidate in candidates:
        observed[candidate.cells[len(candidate.cells) // 2]] = OCCUPIED
    state.update(observed)
    decision = analyze_map_change(state.update(observed), context, planner, scenario.config)
    assert decision.action == PlanningAction.FAST_REPLAN
    assert decision.valid_candidate_count == 0


def test_all_requested_event_scenarios_are_deterministic() -> None:
    expected = {"isolated_one_frame_noise", "persistent_noise", "path_obstacle",
                "far_large_object", "blocked_route_clears", "new_shortcut",
                "stopping_corridor", "outside_safety", "stereo_confirms",
                "stereo_clears", "widespread_corruption", "alternating_flicker",
                "gradual_change"}
    assert expected == set(EVENT_SCENARIO_NAMES)
    for name in sorted(expected):
        first, second = make_event_scenario(name), make_event_scenario(name)
        assert len(first.updates) == len(second.updates) > 0
        np.testing.assert_array_equal(first.updates[0].observed, second.updates[0].observed)
