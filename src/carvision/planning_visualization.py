"""OpenCV diagnostics for local planning and active map inspection."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN
from .planning import Cell, InspectionTarget, PathCandidate, PreparedPlanningGrid


def render_planning_map(
    prepared: PreparedPlanningGrid,
    start: Cell,
    goal: Cell,
    candidates: list[PathCandidate],
    *,
    inspection_target: InspectionTarget | None = None,
    changed_mask: NDArray[np.bool_] | None = None,
    scale: int = 6,
) -> NDArray[np.uint8]:
    if scale < 2:
        raise ValueError("scale must be at least 2")
    data = prepared.data
    image = np.zeros((*data.shape, 3), np.uint8)
    image[data == UNKNOWN] = (125, 125, 125)
    image[data == FREE] = (238, 238, 238)
    image[data == OCCUPIED] = (20, 20, 20)
    inflated_only = prepared.inflated_unsafe & (data != OCCUPIED)
    image[inflated_only] = (20, 120, 240)
    if changed_mask is not None:
        changed = np.asarray(changed_mask, dtype=bool)
        if changed.shape != data.shape:
            raise ValueError("changed_mask shape mismatch")
        image[changed] = (40, 210, 40)
    image = cv2.resize(np.flipud(image), (data.shape[1] * scale, data.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    palette = ((180, 80, 220), (220, 120, 30), (30, 170, 220), (160, 80, 40), (80, 180, 80))
    for index, candidate in reversed(list(enumerate(candidates))):
        points = np.asarray([_pixel(cell, data.shape[0], scale) for cell in candidate.cells], np.int32)
        cv2.polylines(image, [points], False, palette[index % len(palette)], max(1, scale // 3), cv2.LINE_AA)
    if candidates:
        selected = np.asarray([_pixel(cell, data.shape[0], scale) for cell in candidates[0].cells], np.int32)
        cv2.polylines(image, [selected], False, (255, 255, 0), max(2, scale // 2), cv2.LINE_AA)
    if inspection_target is not None:
        center = _pixel(inspection_target.center, data.shape[0], scale)
        cv2.circle(image, center, inspection_target.radius_cells * scale, (220, 0, 220),
                   max(2, scale // 2), cv2.LINE_AA)
        cv2.line(image, _pixel(start, data.shape[0], scale), center, (220, 0, 220), 1, cv2.LINE_AA)
    cv2.circle(image, _pixel(start, data.shape[0], scale), scale, (255, 80, 20), -1)
    cv2.circle(image, _pixel(goal, data.shape[0], scale), scale, (0, 220, 255), -1)
    return _add_legend(image, candidates)


def save_planning_visualization(path: Path | str, *args: object, **kwargs: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render_planning_map(*args, **kwargs))


def _pixel(cell: Cell, height: int, scale: int) -> tuple[int, int]:
    row, column = cell
    return (column * scale + scale // 2, (height - 1 - row) * scale + scale // 2)


def _add_legend(image: NDArray[np.uint8], candidates: list[PathCandidate]) -> NDArray[np.uint8]:
    panel = np.full((76, image.shape[1], 3), 250, np.uint8)
    cv2.putText(panel, "free=white unknown=gray occupied=black inflated=orange changed=green",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, "start=blue goal=yellow selected=cyan inspection=magenta",
                (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    if candidates:
        score = candidates[0].score
        text = (f"best total={score.total:.1f} length={score.length_m:.1f}m "
                f"unknown={score.unknown_fraction:.0%} min_clear={score.minimum_clearance_m:.2f}m")
        cv2.putText(panel, text, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack((panel, image))
