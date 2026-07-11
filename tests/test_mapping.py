import numpy as np
import pytest

from carvision.mapping import FREE, OCCUPIED, UNKNOWN, OccupancyGrid, OccupancyGridConfig


def config() -> OccupancyGridConfig:
    return OccupancyGridConfig(-2.0, 2.0, -1.0, 3.0, 1.0, -0.5, 1.0)


def test_dimensions_and_half_open_boundaries() -> None:
    grid = OccupancyGrid(config())
    assert grid.data.shape == (4, 4)
    assert grid.metric_to_cell(-2.0, -1.0) == (0, 0)
    assert grid.metric_to_cell(1.999, 2.999) == (3, 3)
    assert grid.metric_to_cell(2.0, 0.0) is None
    assert grid.metric_to_cell(0.0, 3.0) is None


def test_ray_marks_free_then_endpoint_occupied() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 2.1, 0.0]]), np.array([0.1, -0.1, 0.0]))
    column = 2
    assert grid.data[3, column] == OCCUPIED
    assert np.all(grid.data[0:3, column] == FREE)
    assert grid.data[0, 0] == UNKNOWN


def test_invalid_height_and_outside_points_are_ignored() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.0, 1.0, 2.0], [5.0, 1.0, 0.0], [np.nan, 1.0, 0.0]]))
    assert np.all(grid.data == UNKNOWN)


def test_invalid_grid_span_rejected() -> None:
    with pytest.raises(ValueError):
        OccupancyGridConfig(0, 1, 0, 1, 0.3)

