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
    assert grid.metric_to_cell(-2.0 - 1e-12, 0.0) is None
    assert grid.metric_to_cell(2.0 - 1e-12, 0.0) == (1, 3)
    assert grid.metric_to_cell(0.0, -1.0 - 1e-12) is None
    assert grid.metric_to_cell(0.0, 3.0 - 1e-12) == (3, 2)
    assert grid.metric_to_cell(np.nan, 0.0) is None


def test_ray_marks_free_then_endpoint_occupied() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 2.1, 0.0]]), np.array([0.1, -0.1, 0.0]))
    column = 2
    assert grid.data[3, column] == OCCUPIED
    assert np.all(grid.data[0:3, column] == FREE)
    assert grid.data[0, 0] == UNKNOWN


def test_invalid_height_and_nonfinite_points_are_ignored() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 0.0]]))
    assert np.all(grid.data == UNKNOWN)


def test_invalid_grid_span_rejected() -> None:
    with pytest.raises(ValueError):
        OccupancyGridConfig(0, 1, 0, 1, 0.3)


def test_ray_with_origin_outside_is_clipped_and_endpoint_occupied() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 2.1, 0.0]]), np.array([0.1, -2.0, 1.2]))
    assert np.all(grid.data[:3, 2] == FREE)
    assert grid.data[3, 2] == OCCUPIED


def test_offset_sensor_origin_does_not_clear_space_behind_mount() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 2.1, 0.0]]), np.array([0.1, 0.5, 1.2]))
    assert grid.data[0, 2] == UNKNOWN
    assert np.all(grid.data[1:3, 2] == FREE)
    assert grid.data[3, 2] == OCCUPIED


def test_return_beyond_grid_clears_crossing_ray_without_false_boundary_obstacle() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 5.0, 0.0]]), np.array([0.1, 0.0, 0.5]))
    assert np.all(grid.data[1:, 2] == FREE)
    assert not np.any(grid.data == OCCUPIED)


def test_both_ends_outside_but_crossing_grid_marks_free() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[3.0, 1.1, 0.0]]), np.array([-3.0, 1.1, 0.0]))
    assert np.all(grid.data[2, :] == FREE)


def test_non_intersecting_outside_ray_is_ignored() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[4.0, 2.0, 0.0]]), np.array([3.0, 2.0, 0.0]))
    assert np.all(grid.data == UNKNOWN)


def test_occupied_dominates_free_conflicts_and_repeated_points() -> None:
    point = np.array([[0.1, 1.1, 0.0]])
    grid = OccupancyGrid(config())
    grid.update(np.repeat(point, 20, axis=0))
    cell = grid.metric_to_cell(0.1, 1.1)
    assert cell is not None and grid.data[cell] == OCCUPIED
    # A later ray passes through that cell; occupied state must survive.
    grid.update(np.array([[0.1, 2.1, 0.0]]))
    assert grid.data[cell] == OCCUPIED


def test_ground_and_extreme_heights_do_not_clear_or_occupy() -> None:
    grid = OccupancyGrid(config())
    grid.update(np.array([[0.1, 2.1, -0.500001], [0.1, 2.1, 1.000001], [0, 1, np.inf]]))
    assert np.all(grid.data == UNKNOWN)
