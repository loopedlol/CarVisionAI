#!/usr/bin/env python3
"""Benchmark planning stages on representative local grid sizes."""

from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

from carvision.mapping import OccupancyGridConfig
from carvision.planning import (
    PlannerConfig, ScoreWeights, apply_simulated_refinement, plan_candidates,
    prepare_planning_grid, select_inspection_targets,
)
from carvision.planning_mock import make_planning_scenario


def main() -> None:
    base = make_planning_scenario("active_inspection")
    planner = PlannerConfig(robot_radius_m=0.2, candidate_count=5,
                            weights=ScoreWeights(1.0, 0.45, 0.55, 1.5, 0.25))
    for scale in (1, 2):
        width, height = base.config.width * scale, base.config.height * scale
        rough = cv2.resize(base.rough_data, (width, height), interpolation=cv2.INTER_NEAREST)
        truth = cv2.resize(base.truth_data, (width, height), interpolation=cv2.INTER_NEAREST)
        config = OccupancyGridConfig(
            base.config.x_min_m, base.config.x_min_m + width * base.config.resolution_m,
            base.config.y_min_m, base.config.y_min_m + height * base.config.resolution_m,
            base.config.resolution_m, base.config.min_height_m, base.config.max_height_m)
        start = (base.start[0] * scale, base.start[1] * scale)
        goal = (base.goal[0] * scale, base.goal[1] * scale)
        stage: dict[str, float] = {}
        begin = perf_counter(); prepared = prepare_planning_grid(rough, config, planner=planner)
        stage["prepare_ms"] = (perf_counter() - begin) * 1000
        begin = perf_counter(); candidates = plan_candidates(prepared, start, goal, planner=planner)
        stage["candidates_ms"] = (perf_counter() - begin) * 1000
        begin = perf_counter(); targets = select_inspection_targets(prepared, candidates, planner=planner, max_targets=1)
        stage["inspection_ms"] = (perf_counter() - begin) * 1000
        begin = perf_counter(); refinement = apply_simulated_refinement(rough, truth, targets[0])
        stage["refinement_ms"] = (perf_counter() - begin) * 1000
        begin = perf_counter(); refined = prepare_planning_grid(refinement.updated_data, config, planner=planner)
        replanned = plan_candidates(refined, start, goal, planner=planner)
        stage["replan_ms"] = (perf_counter() - begin) * 1000
        print(f"grid={height}x{width} candidates={len(candidates)}/{len(replanned)} "
              + " ".join(f"{name}={value:.1f}" for name, value in stage.items())
              + f" total_ms={sum(stage.values()):.1f}")


if __name__ == "__main__":
    main()
