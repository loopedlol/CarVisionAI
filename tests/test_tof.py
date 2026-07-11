import numpy as np
import pytest

from carvision.frames import RigidTransform
from carvision.tof import ranges_to_sensor_points, ranges_to_vehicle_points


def test_tof_axes_and_invalid_range_filtering() -> None:
    points = ranges_to_vehicle_points(
        np.array([2.0, 2.0, np.nan, 20.0]),
        np.array([0.0, np.pi / 2, 0.0, 0.0]),
        max_range_m=10.0,
    )
    np.testing.assert_allclose(points, [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]], atol=1e-12)


def test_tof_elevation() -> None:
    points = ranges_to_vehicle_points(np.array([2.0]), np.array([0.0]), np.array([np.pi / 6]))
    np.testing.assert_allclose(points[0], [0.0, np.sqrt(3.0), 1.0], atol=1e-12)


def test_tof_mount_above_ahead_and_rotated() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mount = RigidTransform(rotation, np.array([0.0, 0.5, 1.0]))
    points = ranges_to_vehicle_points([2.0], [0.0], sensor_to_vehicle=mount)
    np.testing.assert_allclose(points, [[-2.0, 0.5, 1.0]], atol=1e-12)


def test_invalid_and_extreme_ranges() -> None:
    points = ranges_to_sensor_points([0, -1, np.nan, np.inf, 0.05, 10, 10.0001], np.zeros(7),
                                     min_range_m=0.05, max_range_m=10)
    np.testing.assert_allclose(points[:, 1], [0.05, 10.0])
    with pytest.raises(ValueError):
        ranges_to_sensor_points([1], [0], max_range_m=np.nan)
