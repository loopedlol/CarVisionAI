"""Deterministic local-map scenarios for navigation and active inspection."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGridConfig
from .planning import Cell


@dataclass(frozen=True)
class PlanningScenario:
    name: str
    description: str
    config: OccupancyGridConfig
    rough_data: NDArray[np.int8]
    truth_data: NDArray[np.int8]
    start: Cell
    goal: Cell


SCENARIO_NAMES = (
    "open", "blocked_direct", "left_right", "unknown_shortcut", "narrow_opening",
    "false_tof_obstacle", "stereo_confirms_obstacle", "stereo_clears_obstacle",
    "no_route", "invalid_start", "active_inspection",
)


def make_planning_scenario(name: str) -> PlanningScenario:
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name}; choose from {SCENARIO_NAMES}")
    config = OccupancyGridConfig(-4.0, 4.0, 0.0, 10.0, 0.1, -0.4, 1.5)
    rough = _open_map(config)
    truth = rough.copy()
    start, goal = (6, 40), (93, 40)
    description = ""
    if name == "open":
        description = "Open known space with only map boundaries."
    elif name == "blocked_direct":
        rough[44:58, 31:50] = OCCUPIED
        truth[:] = rough
        description = "A solid obstacle blocks the straight route."
    elif name == "left_right":
        rough[38:64, 31:49] = OCCUPIED
        truth[:] = rough
        description = "A central block creates meaningful left and right alternatives."
    elif name == "unknown_shortcut":
        rough[48:52, 4:76] = OCCUPIED
        rough[48:52, 12:22] = FREE
        rough[48:52, 37:44] = UNKNOWN
        truth[:] = rough
        truth[48:52, 37:44] = FREE
        description = "A short unknown opening competes with a long known-safe left route."
    elif name == "narrow_opening":
        rough[49:53, 1:79] = OCCUPIED
        rough[49:53, 36:45] = FREE
        truth[:] = rough
        description = "A single narrow opening tests inflation and clearance cost."
    elif name == "false_tof_obstacle":
        rough[49, 40] = OCCUPIED
        truth[:] = rough
        truth[49, 40] = FREE
        description = "One isolated ToF return falsely blocks the direct centerline."
    elif name == "stereo_confirms_obstacle":
        rough[45:55, 36:45] = UNKNOWN
        truth[:] = rough
        truth[45:55, 36:45] = OCCUPIED
        description = "Stereo confirms that an uncertain central region is occupied."
    elif name == "stereo_clears_obstacle":
        rough[48:52, 38:43] = OCCUPIED
        truth[:] = rough
        truth[48:52, 38:43] = FREE
        description = "Stereo clears an apparent compact ToF obstacle."
    elif name == "no_route":
        rough[49:54, 1:79] = OCCUPIED
        truth[:] = rough
        description = "A boundary-to-boundary wall makes the goal unreachable."
    elif name == "invalid_start":
        rough[start] = OCCUPIED
        truth[:] = rough
        description = "The requested start lies in an occupied cell."
    elif name == "active_inspection":
        rough[47:52, 1:79] = OCCUPIED
        rough[47:52, 10:21] = FREE
        rough[47:52, 37:44] = UNKNOWN
        rough[42, 47] = OCCUPIED  # suspicious isolated 360-degree ToF return
        truth[:] = rough
        truth[47:52, 37:44] = OCCUPIED
        truth[42, 47] = FREE
        description = "A short uncertain opening is later confirmed blocked; a false isolated return is cleared."
    return PlanningScenario(name, description, config, rough, truth, start, goal)


def _open_map(config: OccupancyGridConfig) -> NDArray[np.int8]:
    data = np.full((config.height, config.width), FREE, dtype=np.int8)
    data[[0, -1], :] = OCCUPIED
    data[:, [0, -1]] = OCCUPIED
    return data
