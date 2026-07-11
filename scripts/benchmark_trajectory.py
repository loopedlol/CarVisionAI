#!/usr/bin/env python3
"""Benchmark trajectory-only and path-plus-trajectory operations."""

from statistics import median
from time import perf_counter

from carvision.mapping import OCCUPIED
from carvision.planning import PlannerConfig, plan_candidates, prepare_planning_grid
from carvision.planning_mock import make_planning_scenario
from carvision.trajectory import TrajectoryConfig, generate_trajectory, validate_trajectory


def main() -> None:
    scenario = make_planning_scenario("blocked_direct")
    planner = PlannerConfig(robot_radius_m=0.2, candidate_count=5)
    config = TrajectoryConfig(robot_radius_m=0.2, safety_margin_m=0.05, sample_spacing_m=0.03)
    prepared = prepare_planning_grid(scenario.rough_data, scenario.config, planner=planner)
    candidates = plan_candidates(prepared, scenario.start, scenario.goal, planner=planner)
    trajectory = generate_trajectory(candidates[0], prepared, config)
    updated = scenario.rough_data.copy()
    sample = trajectory.samples[min(12, len(trajectory.samples) - 1)]
    cell = (int((sample.y_m - scenario.config.y_min_m) / scenario.config.resolution_m),
            int((sample.x_m - scenario.config.x_min_m) / scenario.config.resolution_m))
    updated[cell] = OCCUPIED
    metrics: dict[str, list[float]] = {name: [] for name in (
        "generation", "regeneration", "validation", "stopping_validation",
        "fast_planning", "fast_total", "full_planning", "full_total")}
    stage: dict[str, list[float]] = {}
    for _ in range(5):
        begin = perf_counter(); generated = generate_trajectory(candidates[0], prepared, config)
        metrics["generation"].append((perf_counter() - begin) * 1000)
        for name, value in generated.timings_ms.items(): stage.setdefault(name, []).append(value)
        begin = perf_counter(); generate_trajectory(candidates[0], prepared, config)
        metrics["regeneration"].append((perf_counter() - begin) * 1000)
        begin = perf_counter(); validate_trajectory(trajectory, scenario.rough_data,
                                                    scenario.config, planner, config)
        metrics["validation"].append((perf_counter() - begin) * 1000)
        begin = perf_counter(); validate_trajectory(trajectory, updated, scenario.config,
                                                    planner, config, current_speed_mps=0.7)
        metrics["stopping_validation"].append((perf_counter() - begin) * 1000)

        fast = PlannerConfig(**{**planner.__dict__, "candidate_count": 1, "candidate_attempts": 1})
        begin = perf_counter(); fast_candidates = plan_candidates(prepared, scenario.start,
                                                                  scenario.goal, planner=fast)
        plan_ms = (perf_counter() - begin) * 1000; metrics["fast_planning"].append(plan_ms)
        begin = perf_counter(); generate_trajectory(fast_candidates[0], prepared, config)
        metrics["fast_total"].append(plan_ms + (perf_counter() - begin) * 1000)

        begin = perf_counter(); full_candidates = plan_candidates(prepared, scenario.start,
                                                                  scenario.goal, planner=planner)
        plan_ms = (perf_counter() - begin) * 1000; metrics["full_planning"].append(plan_ms)
        begin = perf_counter(); generate_trajectory(full_candidates[0], prepared, config)
        metrics["full_total"].append(plan_ms + (perf_counter() - begin) * 1000)
    print(" ".join(f"{name}_median_ms={median(values):.2f}" for name, values in metrics.items()))
    print("generation_stages " + " ".join(
        f"{name}_median_ms={median(values):.2f}" for name, values in stage.items()))


if __name__ == "__main__": main()
