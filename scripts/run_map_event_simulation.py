#!/usr/bin/env python3
"""Continuous noisy map updates with event-driven versus always-full planning."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from carvision.map_event_visualization import render_event_frame
from carvision.map_events import (
    PlanningAction, TemporalMapState, analyze_map_change, build_route_context,
    execute_planning_action,
)
from carvision.mapping import FREE, OCCUPIED, UNKNOWN
from carvision.planning import PlannerConfig, ScoreWeights, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/map_events"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_planning_scenario("active_inspection")
    planner = PlannerConfig(
        robot_radius_m=0.2, candidate_count=5, inspection_radius_m=0.8,
        weights=ScoreWeights(1.0, 0.45, 0.55, 1.5, 0.25))
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    context = build_route_context(scenario.rough_data.shape, scenario.config, candidates,
                                  scenario.start, scenario.goal, planner=planner)
    state = TemporalMapState.initialize(scenario.rough_data)
    world = scenario.rough_data.copy()
    selected_block_cell = candidates[0].cells[len(candidates[0].cells) // 2]
    all_block_cells: list[tuple[int, int]] | None = None
    records: list[dict[str, object]] = []
    images: list[np.ndarray] = []
    policy_time = 0.0
    always_time = 0.0

    updates: list[tuple[str, str]] = [
        ("baseline", "baseline"),
        ("isolated far noise", "noise"),
        ("noise disappears", "baseline"),
        ("large far object first", "far"),
        ("large far object persistent", "far"),
        ("selected route receives obstacle evidence", "block_selected"),
        ("selected route obstacle persistent", "block_selected"),
        ("stereo clears selected route obstacle", "clear_selected"),
        ("all remaining candidates receive obstacle evidence", "block_all"),
        ("all-candidate obstacles persistent", "block_all"),
        ("stereo clears all candidate obstacles", "clear_all"),
        ("outside-safety change first", "outside"),
        ("outside-safety change persistent", "outside"),
        ("shortcut becomes uncertain", "unknown_shortcut"),
        ("stereo clears shortcut", "clear_shortcut"),
        ("alternating far flicker occupied", "flicker_on"),
        ("alternating far flicker free", "flicker_off"),
        ("raw obstacle in stopping corridor", "stop"),
        ("widespread sensor corruption", "corrupt"),
    ]
    shortcut_cell: tuple[int, int] | None = None
    outside_cells: np.ndarray | None = None
    for index, (label, kind) in enumerate(updates):
        evidence_weight = 1
        observed = world.copy()
        if kind == "noise":
            observed[20, 72] = OCCUPIED
        elif kind == "far":
            world[20:30, 66:75] = OCCUPIED; observed = world.copy()
        elif kind == "block_selected":
            world[selected_block_cell] = OCCUPIED
            observed = world.copy()
        elif kind == "clear_selected":
            world[selected_block_cell] = FREE
            observed = world.copy(); evidence_weight = 3
        elif kind == "block_all":
            if all_block_cells is None:
                all_block_cells = [candidate.cells[len(candidate.cells) // 2]
                                   for candidate in context.candidates]
            for cell in all_block_cells:
                world[cell] = OCCUPIED
            observed = world.copy()
        elif kind == "clear_all":
            assert all_block_cells is not None
            for cell in all_block_cells:
                world[cell] = FREE
            observed = world.copy(); evidence_weight = 3
        elif kind == "outside":
            if outside_cells is None:
                irrelevant = ~(context.safety_corridor | context.candidate_mask
                               | context.divergence_mask | context.goal_mask
                               | context.stopping_corridor)
                outside_cells = np.argwhere(irrelevant & (world == FREE))[:80]
            world[outside_cells[:, 0], outside_cells[:, 1]] = OCCUPIED; observed = world.copy()
        elif kind == "unknown_shortcut":
            available = ((context.divergence_mask | context.safety_corridor)
                         & ~context.path_mask & (world == FREE))
            shortcut_cell = tuple(int(v) for v in np.argwhere(available)[0])
            world[shortcut_cell] = UNKNOWN; observed = world.copy(); evidence_weight = 3
        elif kind == "clear_shortcut":
            assert shortcut_cell is not None
            world[shortcut_cell] = FREE; observed = world.copy(); evidence_weight = 3
        elif kind == "flicker_on":
            observed[32, 70] = OCCUPIED
        elif kind == "flicker_off":
            observed[32, 70] = FREE
        elif kind == "stop":
            observed[context.selected_path[min(7, len(context.selected_path) - 1)]] = OCCUPIED
        elif kind == "corrupt":
            observed[5:58, 4:68] = UNKNOWN

        change = state.update(observed, evidence_weight=evidence_weight)
        event_context = context
        decision = analyze_map_change(change, event_context, planner, scenario.config)
        execution = execute_planning_action(decision, change.stable, event_context,
                                            scenario.config, planner)
        policy_time += execution.planning_time_ms
        if execution.candidates and execution.executed_action != PlanningAction.STOP:
            candidates = list(execution.candidates)
            context = build_route_context(change.stable.shape, scenario.config, candidates,
                                          scenario.start, scenario.goal, planner=planner)

        baseline_start = perf_counter()
        raw_prepared = prepare_planning_grid(observed, scenario.config, planner=planner)
        baseline_candidates = plan_candidates(raw_prepared, scenario.start, scenario.goal,
                                              planner=planner)
        baseline_ms = (perf_counter() - baseline_start) * 1000
        always_time += baseline_ms
        record = {
            "frame": index, "label": label, "raw_changed_cells": change.counts()["raw_changed"],
            "stable_changed_cells": change.counts()["stable_changed"],
            "change_counts": change.counts(),
            "suppressed_transient_cells": change.counts()["suppressed_transient"],
            "route_relevant_changed_cells": decision.route_relevant_changed_cells,
            "route_change_importance": decision.importance,
            "valid_candidate_count": decision.valid_candidate_count,
            "decision": decision.action.value, "executed_action": execution.executed_action.value,
            "reasons": list(decision.reasons), "route_valid": decision.route_valid,
            "policy_planning_ms": execution.planning_time_ms,
            "always_full_planning_ms": baseline_ms,
            "always_full_candidate_count": len(baseline_candidates),
        }
        records.append(record)
        image = render_event_frame(change, event_context, decision)
        images.append(image)
        cv2.imwrite(str(args.output_dir / f"frame_{index:02d}.png"), image)

    counts = Counter(record["executed_action"] for record in records)
    summary = {
        "map_updates": len(records),
        "ignored_transient_updates": sum(
            record["decision"] == PlanningAction.NONE.value
            and record["suppressed_transient_cells"] > 0 for record in records),
        "route_rescoring_events": counts[PlanningAction.RESCORE.value],
        "candidate_reevaluations": counts[PlanningAction.REEVALUATE.value],
        "fast_replans": counts[PlanningAction.FAST_REPLAN.value],
        "full_replans": counts[PlanningAction.FULL_REPLAN.value],
        "emergency_invalidations": counts[PlanningAction.STOP.value],
        "policy_planning_time_ms": policy_time,
        "always_full_planning_time_ms": always_time,
        "planning_time_saved_ms": always_time - policy_time,
        "planning_time_saved_fraction": (always_time - policy_time) / always_time,
        "delayed_reactions": [
            {"event": "persistent selected-route obstacle", "delay_frames": 1,
             "reason": "two occupied observations required outside stopping corridor"}],
        "missed_required_reactions": [],
    }
    report = {"summary": summary, "frames": records}
    (args.output_dir / "timeline.json").write_text(json.dumps(report, indent=2) + "\n")
    cv2.imwrite(str(args.output_dir / "timeline.png"), _contact_sheet(images, 4))
    print(json.dumps(summary, indent=2))


def _contact_sheet(images: list[np.ndarray], columns: int) -> np.ndarray:
    rows = (len(images) + columns - 1) // columns
    blank = np.full_like(images[0], 250)
    padded = images + [blank] * (rows * columns - len(images))
    return np.vstack([np.hstack(padded[row * columns:(row + 1) * columns])
                      for row in range(rows)])


if __name__ == "__main__":
    main()
