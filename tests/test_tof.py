import numpy as np

from carvision.tof import ranges_to_vehicle_points


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

