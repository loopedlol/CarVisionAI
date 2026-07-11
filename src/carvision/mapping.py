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
        values = (
            self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m,
            self.resolution_m, self.min_height_m, self.max_height_m,
        )
        if not np.isfinite(values).all():
            raise ValueError("grid configuration must be finite")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("grid maximums must exceed minimums")
        if self.resolution_m <= 0 or self.max_height_m < self.min_height_m:
            raise ValueError("resolution must be positive and height bounds ordered")
        for span in (self.x_max_m - self.x_min_m, self.y_max_m - self.y_min_m):
            cells = span / self.resolution_m
            if not np.isclose(cells, round(cells), atol=1e-9):
                raise ValueError("grid spans must be integer multiples of resolution")

    @property
    def width(self) -> int:
        return round((self.x_max_m - self.x_min_m) / self.resolution_m)

    @property
    def height(self) -> int:
        return round((self.y_max_m - self.y_min_m) / self.resolution_m)


class OccupancyGrid:
    """A local ternary occupancy grid populated from vehicle-frame endpoints."""

    def __init__(self, config: OccupancyGridConfig) -> None:
        self.config = config
        self.data = np.full((config.height, config.width), UNKNOWN, dtype=np.int8)

    def metric_to_cell(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        """Return (row, column), or None outside half-open grid bounds."""
        c = self.config
        if not (c.x_min_m <= x_m < c.x_max_m and c.y_min_m <= y_m < c.y_max_m):
            return None
        column = int(np.floor((x_m - c.x_min_m) / c.resolution_m))
        row = int(np.floor((y_m - c.y_min_m) / c.resolution_m))
        return row, column

    def update(
        self,
        points_vehicle_m: NDArray[np.floating],
        sensor_origin_m: NDArray[np.floating] = np.zeros(3),
        *,
        mark_free: bool = True,
    ) -> None:
        """Ray-trace valid endpoints, marking free cells then occupied endpoints.

        Points outside the XY map are ignored rather than treated as occupied at
        the boundary. Height filtering applies to endpoints, not the rays.
        """
        points = np.asarray(points_vehicle_m, dtype=np.float64)
        origin = np.asarray(sensor_origin_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_vehicle_m must have shape (N, 3)")
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("sensor_origin_m must be one finite 3D point")
        origin_cell = self.metric_to_cell(float(origin[0]), float(origin[1]))
        for point in points:
            if not np.isfinite(point).all():
                continue
            if not self.config.min_height_m <= point[2] <= self.config.max_height_m:
                continue
            endpoint = self.metric_to_cell(float(point[0]), float(point[1]))
            if endpoint is None:
                continue
            if mark_free and origin_cell is not None:
                ray = _bresenham(origin_cell, endpoint)
                for row, column in ray[:-1]:
                    if self.data[row, column] != OCCUPIED:
                        self.data[row, column] = FREE
            self.data[endpoint] = OCCUPIED

    def preview_u8(self) -> NDArray[np.uint8]:
        """Return a display image: unknown 127, free 32, occupied 255."""
        image = np.full(self.data.shape, 127, dtype=np.uint8)
        image[self.data == FREE] = 32
        image[self.data == OCCUPIED] = 255
        return np.flipud(image)


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Integer grid cells on a line, including both endpoints."""
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

