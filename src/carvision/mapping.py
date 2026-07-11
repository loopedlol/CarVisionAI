"""Sensor-independent local occupancy mapping."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

UNKNOWN: np.int8 = np.int8(-1)
FREE: np.int8 = np.int8(0)
OCCUPIED: np.int8 = np.int8(100)


@dataclass(frozen=True)
class OccupancyGridConfig:
    x_min_m: float = -5.0
    x_max_m: float = 5.0
    y_min_m: float = 0.0
    y_max_m: float = 10.0
    resolution_m: float = 0.1
    min_height_m: float = -0.5
    max_height_m: float = 1.5

    def __post_init__(self) -> None:
        values = (self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m,
                  self.resolution_m, self.min_height_m, self.max_height_m)
        if not np.isfinite(values).all():
            raise ValueError("grid configuration must be finite")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("grid maximums must exceed minimums")
        if self.resolution_m <= 0 or self.max_height_m < self.min_height_m:
            raise ValueError("resolution must be positive and height bounds ordered")
        for span in (self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m):
            if not np.isclose(span / self.resolution_m, round(span / self.resolution_m), atol=1e-9):
                raise ValueError("grid spans must be integer multiples of resolution")

    @property
    def width(self) -> int:
        return round((self.x_max_m - self.x_min_m) / self.resolution_m)

    @property
    def height(self) -> int:
        return round((self.y_max_m - self.y_min_m) / self.resolution_m)


class OccupancyGrid:
    """Local ternary grid; occupied evidence always dominates free evidence."""

    def __init__(self, config: OccupancyGridConfig) -> None:
        self.config = config
        self.data = np.full((config.height, config.width), UNKNOWN, dtype=np.int8)

    def metric_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        """Return (row, column), or None outside finite half-open bounds."""
        c = self.config
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            return None
        if not (c.x_min_m <= x_m < c.x_max_m and c.y_min_m <= y_m < c.y_max_m):
            return None
        # Cap only after the metric half-open check. Floating arithmetic at the
        # last representable interior value can otherwise round the quotient to
        # exactly width/height and create an out-of-bounds index.
        row = min(int(np.floor((y_m - c.y_min_m) / c.resolution_m)), c.height - 1)
        column = min(int(np.floor((x_m - c.x_min_m) / c.resolution_m)), c.width - 1)
        return row, column

    def update(
        self,
        points_vehicle_m: NDArray[np.floating],
        sensor_origin_m: NDArray[np.floating] | None = None,
        *,
        mark_free: bool = True,
    ) -> None:
        """Integrate endpoint returns and clipped free-space rays.

        Endpoint heights outside the configured obstacle band are ignored
        completely: they neither mark occupied cells nor clear free space. Rays
        are clipped to the XY grid, allowing either end to lie outside. Repeated
        observations are collapsed at grid-cell level before tracing.
        """
        points = np.asarray(points_vehicle_m, dtype=np.float64)
        origin = np.zeros(3) if sensor_origin_m is None else np.asarray(sensor_origin_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_vehicle_m must have shape (N, 3)")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("sensor_origin_m must be one finite 3D point")

        c = self.config
        valid = (np.isfinite(points).all(axis=1)
                 & (points[:, 2] >= c.min_height_m)
                 & (points[:, 2] <= c.max_height_m))
        filtered = points[valid]
        inside = ((filtered[:, 0] >= c.x_min_m) & (filtered[:, 0] < c.x_max_m)
                  & (filtered[:, 1] >= c.y_min_m) & (filtered[:, 1] < c.y_max_m))
        observations: dict[tuple[int, int], bool] = {}
        if np.any(inside):
            interior = filtered[inside]
            columns = np.floor((interior[:, 0] - c.x_min_m) / c.resolution_m).astype(np.int64)
            rows = np.floor((interior[:, 1] - c.y_min_m) / c.resolution_m).astype(np.int64)
            columns = np.minimum(columns, c.width - 1)
            rows = np.minimum(rows, c.height - 1)
            for row, column in np.unique(np.column_stack((rows, columns)), axis=0):
                observations[(int(row), int(column))] = True

        # Only out-of-grid returns need continuous segment clipping. Dense
        # in-grid clouds have already been collapsed to unique occupied cells.
        for point in filtered[~inside]:
            clipped = _clip_segment_to_grid(origin[:2], point[:2], c)
            if clipped is None:
                continue
            terminal = self.metric_to_cell(float(clipped[1][0]), float(clipped[1][1]))
            if terminal is not None:
                observations.setdefault(terminal, False)

        occupied_cells = {cell for cell, occupied in observations.items() if occupied}
        if mark_free:
            for terminal, is_occupied in observations.items():
                clipped = _clip_segment_to_grid(origin[:2], _cell_center(terminal, c), c)
                if clipped is None:
                    continue
                start = self.metric_to_cell(*clipped[0])
                end = self.metric_to_cell(*clipped[1])
                if start is None or end is None:
                    continue
                ray = _bresenham(start, end)
                free_cells = ray[:-1] if is_occupied else ray
                for row, column in free_cells:
                    if (row, column) not in occupied_cells and self.data[row, column] != OCCUPIED:
                        self.data[row, column] = FREE
        for row, column in occupied_cells:
            self.data[row, column] = OCCUPIED

    def preview_u8(self) -> NDArray[np.uint8]:
        image = np.full(self.data.shape, 127, dtype=np.uint8)
        image[self.data == FREE] = 32
        image[self.data == OCCUPIED] = 255
        return np.flipud(image)


def _cell_center(cell: tuple[int, int], c: OccupancyGridConfig) -> NDArray[np.float64]:
    row, column = cell
    return np.array([c.x_min_m + (column + 0.5) * c.resolution_m,
                     c.y_min_m + (row + 0.5) * c.resolution_m])


def _clip_segment_to_grid(
    start_xy: NDArray[np.floating], end_xy: NDArray[np.floating], c: OccupancyGridConfig
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clip to the grid, nudged inside its half-open maxima."""
    start = np.asarray(start_xy, dtype=np.float64)
    end = np.asarray(end_xy, dtype=np.float64)
    delta = end - start
    t0, t1 = 0.0, 1.0
    bounds = ((c.x_min_m, np.nextafter(c.x_max_m, -np.inf)),
              (c.y_min_m, np.nextafter(c.y_max_m, -np.inf)))
    for axis, (lower, upper) in enumerate(bounds):
        if delta[axis] == 0.0:
            if start[axis] < lower or start[axis] > upper:
                return None
            continue
        enter = (lower - start[axis]) / delta[axis]
        leave = (upper - start[axis]) / delta[axis]
        if enter > leave:
            enter, leave = leave, enter
        t0, t1 = max(t0, enter), min(t1, leave)
        if t0 > t1:
            return None
    clipped_start = start + t0 * delta
    clipped_end = start + t1 * delta
    # Recomposition can round nextafter(max, -inf) back to max for long
    # segments, so enforce representable interior coordinates explicitly.
    clipped_start = np.array([
        np.clip(clipped_start[0], bounds[0][0], bounds[0][1]),
        np.clip(clipped_start[1], bounds[1][0], bounds[1][1]),
    ])
    clipped_end = np.array([
        np.clip(clipped_end[0], bounds[0][0], bounds[0][1]),
        np.clip(clipped_end[1], bounds[1][0], bounds[1][1]),
    ])
    return ((float(clipped_start[0]), float(clipped_start[1])),
            (float(clipped_end[0]), float(clipped_end[1])))


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    y0, x0 = start
    y1, x1 = end
    cells: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            return cells
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy
