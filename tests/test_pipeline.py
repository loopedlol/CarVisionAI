import numpy as np

from carvision.frames import camera_to_vehicle
from carvision.mapping import OCCUPIED, OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_mock_stereo_scene, make_mock_tof_scan
from carvision.stereo import backproject_depth, disparity_to_depth, valid_points
from carvision.tof import ranges_to_vehicle_points


def test_mock_stereo_pipeline_geometry() -> None:
    scene = make_mock_stereo_scene()
    depth = disparity_to_depth(scene.disparity_px, scene.calibration)
    np.testing.assert_allclose(depth, scene.expected_depth_m, equal_nan=True)
    points = valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration)))
    assert len(points) > 0
    assert np.isclose(points[:, 1].min(), 2.5)
    assert np.isclose(points[:, 1].max(), 6.0)


def test_both_sensor_types_feed_same_mapper() -> None:
    scene = make_mock_stereo_scene()
    stereo = valid_points(camera_to_vehicle(backproject_depth(
        disparity_to_depth(scene.disparity_px, scene.calibration), scene.calibration
    )))
    ranges, azimuth = make_mock_tof_scan()
    tof = ranges_to_vehicle_points(ranges, azimuth, max_range_m=10.0)
    grid = OccupancyGrid(OccupancyGridConfig(y_min_m=-1, y_max_m=8))
    grid.update(np.concatenate((stereo, tof)))
    assert np.count_nonzero(grid.data == OCCUPIED) > 20
    assert np.count_nonzero(grid.data == 0) > 20
