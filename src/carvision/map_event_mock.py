"""Synthetic occupancy update streams for temporal trigger validation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN
from .planning_mock import PlanningScenario, make_planning_scenario


@dataclass(frozen=True)
class MockMapUpdate:
    label: str
    observed: NDArray[np.int8]
    evidence_weight: int = 1
    observation_confidence: float = 1.0


@dataclass(frozen=True)
class MapEventScenario:
    name: str
    base: PlanningScenario
    updates: tuple[MockMapUpdate, ...]


EVENT_SCENARIO_NAMES = (
    "isolated_one_frame_noise", "persistent_noise", "path_obstacle", "far_large_object",
    "blocked_route_clears", "new_shortcut", "stopping_corridor", "outside_safety",
    "stereo_confirms", "stereo_clears", "widespread_corruption", "alternating_flicker",
    "gradual_change",
)


def make_event_scenario(name: str) -> MapEventScenario:
    if name not in EVENT_SCENARIO_NAMES:
        raise ValueError(f"unknown event scenario {name}")
    base_name = "open"
    if name == "blocked_route_clears":
        base_name = "blocked_direct"
    elif name == "new_shortcut":
        base_name = "unknown_shortcut"
    elif name == "stereo_confirms":
        base_name = "stereo_confirms_obstacle"
    elif name == "stereo_clears":
        base_name = "stereo_clears_obstacle"
    base = make_planning_scenario(base_name)
    stable = base.rough_data.copy()
    updates: list[MockMapUpdate] = []
    if name == "isolated_one_frame_noise":
        noisy = stable.copy(); noisy[20, 70] = OCCUPIED
        updates = [MockMapUpdate("isolated noise", noisy), MockMapUpdate("noise disappears", stable)]
    elif name == "persistent_noise":
        noisy = stable.copy(); noisy[20, 70] = OCCUPIED
        updates = [MockMapUpdate("first return", noisy), MockMapUpdate("persistent return", noisy)]
    elif name == "path_obstacle":
        blocked = stable.copy(); blocked[45, 40] = OCCUPIED
        updates = [MockMapUpdate("first path return", blocked), MockMapUpdate("confirmed path return", blocked)]
    elif name == "far_large_object":
        far = stable.copy(); far[20:32, 62:75] = OCCUPIED
        updates = [MockMapUpdate("far object first", far), MockMapUpdate("far object persistent", far)]
    elif name == "blocked_route_clears":
        clear = stable.copy(); clear[44:58, 31:50] = FREE
        updates = [MockMapUpdate(f"clear evidence {index}", clear) for index in range(3)]
    elif name == "new_shortcut":
        clear = stable.copy(); clear[48:52, 37:44] = FREE
        updates = [MockMapUpdate("stereo clears shortcut", clear, evidence_weight=3)]
    elif name == "stopping_corridor":
        obstacle = stable.copy(); obstacle[14, 40] = OCCUPIED
        updates = [MockMapUpdate("raw stopping obstacle", obstacle)]
    elif name == "outside_safety":
        obstacle = stable.copy(); obstacle[35:40, 53:58] = OCCUPIED
        updates = [MockMapUpdate("outside corridor first", obstacle),
                   MockMapUpdate("outside corridor persistent", obstacle)]
    elif name == "stereo_confirms":
        confirmed = stable.copy(); confirmed[45:55, 36:45] = OCCUPIED
        updates = [MockMapUpdate("stereo confirms unknown", confirmed, evidence_weight=3)]
    elif name == "stereo_clears":
        cleared = stable.copy(); cleared[48:52, 38:43] = FREE
        updates = [MockMapUpdate("stereo clears false return", cleared, evidence_weight=3)]
    elif name == "widespread_corruption":
        corrupted = stable.copy(); corrupted[5:55, 5:70] = UNKNOWN
        updates = [MockMapUpdate("widespread corruption", corrupted)]
    elif name == "alternating_flicker":
        occupied = stable.copy(); occupied[30, 65] = OCCUPIED
        updates = tuple(MockMapUpdate(f"flicker {index}", occupied if index % 2 == 0 else stable)
                        for index in range(6))
    elif name == "gradual_change":
        current = stable.copy()
        for index in range(4):
            current = current.copy(); current[35 + index, 55:58] = OCCUPIED
            updates.append(MockMapUpdate(f"gradual growth {index} first", current))
            updates.append(MockMapUpdate(f"gradual growth {index} persistent", current))
    return MapEventScenario(name, base, tuple(updates))

