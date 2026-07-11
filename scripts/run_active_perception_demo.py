#!/usr/bin/env python3
"""End-to-end rough-map, inspect, refine, and replan demonstration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from carvision.planning import (
    PlannerConfig, ScoreWeights, apply_simulated_refinement, plan_candidates,
    prepare_planning_grid, select_inspection_targets,
)
from carvision.planning_mock import make_planning_scenario
from carvision.planning_visualization import render_planning_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/planning_demo"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = make_planning_scenario("active_inspection")
    planner = PlannerConfig(
        robot_radius_m=0.2, unknown_policy="penalize", candidate_count=5,
        inspection_radius_m=0.8,
        weights=ScoreWeights(length=1.0, clearance=0.45, unknown=0.55,
                             narrow=1.5, heading_change=0.25),
    )
    rough = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    initial = plan_candidates(rough, scenario.start, scenario.goal, planner=planner)
    if not initial:
        raise RuntimeError("demo produced no initial route")
    targets = select_inspection_targets(rough, initial, planner=planner, max_targets=1)
    if not targets:
        raise RuntimeError("demo produced no inspection target")
    target = targets[0]
    refinement = apply_simulated_refinement(scenario.rough_data, scenario.truth_data, target)
    refined = prepare_planning_grid(refinement.updated_data, scenario.config, planner=planner)
    replanned = plan_candidates(refined, scenario.start, scenario.goal, planner=planner)
    if not replanned:
        raise RuntimeError("demo produced no route after refinement")

    images = {
        "01_rough_candidates.png": render_planning_map(rough, scenario.start, scenario.goal, initial),
        "02_inspection_target.png": render_planning_map(
            rough, scenario.start, scenario.goal, initial, inspection_target=target),
        "03_refined_changed_cells.png": render_planning_map(
            refined, scenario.start, scenario.goal, initial,
            inspection_target=target, changed_mask=refinement.changed_mask),
        "04_replanned.png": render_planning_map(
            refined, scenario.start, scenario.goal, replanned,
            inspection_target=target, changed_mask=refinement.changed_mask),
    }
    for name, image in images.items():
        cv2.imwrite(str(args.output_dir / name), image)
    cv2.imwrite(str(args.output_dir / "active_perception_sequence.png"),
                np.hstack(tuple(images.values())))
    report = {
        "scenario": scenario.name, "description": scenario.description,
        "planner": asdict(planner), "inspection_target": asdict(target),
        "changed_cell_count": int(np.count_nonzero(refinement.changed_mask)),
        "initial_candidates": [_candidate_dict(candidate) for candidate in initial],
        "replanned_candidates": [_candidate_dict(candidate) for candidate in replanned],
        "selected_path_changed": initial[0].cells != replanned[0].cells,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"initial={len(initial)} target={target.center} reasons={target.reasons} "
          f"changed={report['changed_cell_count']} replanned={len(replanned)} "
          f"path_changed={report['selected_path_changed']}")


def _candidate_dict(candidate: object) -> dict[str, object]:
    return {"rank": candidate.rank, "cell_count": len(candidate.cells),
            "score": asdict(candidate.score)}


if __name__ == "__main__":
    main()

