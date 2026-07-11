"""Deterministic maps and grid routes for trajectory validation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGridConfig
from .planning import Cell


@dataclass(frozen=True)
class TrajectoryScenario:
    name: str
    config: OccupancyGridConfig
    data: NDArray[np.int8]
    path: tuple[Cell, ...]
    final_heading_rad: float | None = None


TRAJECTORY_SCENARIOS = (
    "straight", "corner_90", "diagonal_zigzag", "tight_s", "narrow_corridor",
    "inside_corner_obstacle", "unknown_near_route", "clearance_transition",
    "braking_obstacle", "beyond_braking", "trajectory_only_regeneration",
    "immediate_stop", "unsafe_smoothing", "fallback", "final_heading",
)


def make_trajectory_scenario(name: str) -> TrajectoryScenario:
    if name not in TRAJECTORY_SCENARIOS:
        raise ValueError(f"unknown trajectory scenario {name}")
    config = OccupancyGridConfig(-4, 4, 0, 8, 0.1)
    data = np.full((80, 80), FREE, np.int8)
    data[[0, -1], :] = OCCUPIED; data[:, [0, -1]] = OCCUPIED
    path: tuple[Cell, ...]
    final = None
    if name == "straight":
        path = tuple((row, 40) for row in range(6, 74))
    elif name == "corner_90":
        path = tuple((row, 25) for row in range(8, 45)) + tuple((44, col) for col in range(26, 61))
        data[15:42, 28:58] = OCCUPIED
    elif name == "diagonal_zigzag":
        path = tuple((8 + index, 15 + index) for index in range(50))
    elif name == "tight_s":
        path = (_segment((8, 20), (30, 20)) + _segment((30, 20), (30, 55))[1:]
                + _segment((30, 55), (55, 55))[1:] + _segment((55, 55), (55, 25))[1:]
                + _segment((55, 25), (72, 25))[1:])
    elif name == "narrow_corridor":
        data[5:75, 34] = OCCUPIED; data[5:75, 46] = OCCUPIED
        path = tuple((row, 40) for row in range(8, 72))
    elif name in ("inside_corner_obstacle", "unsafe_smoothing", "fallback"):
        path = tuple((row, 25) for row in range(8, 45)) + tuple((44, col) for col in range(26, 61))
        data[15:42, 28:58] = OCCUPIED
        data[42, 27] = OCCUPIED
    elif name == "unknown_near_route":
        path = tuple((row, 40) for row in range(6, 74)); data[35:50, 39:42] = UNKNOWN
    elif name == "clearance_transition":
        path = tuple((row, 40) for row in range(6, 74)); data[42:72, 35] = OCCUPIED
    elif name in ("braking_obstacle", "immediate_stop"):
        path = tuple((row, 40) for row in range(6, 74)); data[20:23, 39:42] = OCCUPIED
    elif name == "beyond_braking":
        path = tuple((row, 40) for row in range(6, 74)); data[62:65, 39:42] = OCCUPIED
    elif name == "trajectory_only_regeneration":
        path = tuple((row, 25) for row in range(8, 45)) + tuple((44, col) for col in range(26, 61))
        data[15:42, 28:58] = OCCUPIED
    elif name == "final_heading":
        path = tuple((row, 40) for row in range(6, 74)); final = np.pi / 2
    return TrajectoryScenario(name, config, data, path, final)


def _segment(start: Cell, end: Cell) -> tuple[Cell, ...]:
    row0, col0 = start; row1, col1 = end
    count = max(abs(row1 - row0), abs(col1 - col0))
    return tuple((round(row0 + (row1 - row0) * index / count),
                  round(col0 + (col1 - col0) * index / count))
                 for index in range(count + 1))
