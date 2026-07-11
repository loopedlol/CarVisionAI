import numpy as np

from carvision.frames import camera_to_vehicle
from carvision.mapping import OCCUPIED, OccupancyGrid, OccupancyGridConfig
from carvision.mock import make_mock_stereo_scene, make_mock_tof_scan, make_noisy_multisensor_scene
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


def test_noisy_stereo_and_tof_overlap_on_known_obstacle_cells() -> None:
    scene = make_noisy_multisensor_scene(seed=11)
    depth = disparity_to_depth(scene.disparity_px, scene.calibration, max_depth_m=12)
    stereo = valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration)))
    tof = ranges_to_vehicle_points(scene.tof_ranges_m, scene.tof_azimuth_rad, max_range_m=12)
    config = OccupancyGridConfig(-6, 6, -1, 9, 0.1, -0.4, 1.4)
    stereo_grid, tof_grid = OccupancyGrid(config), OccupancyGrid(config)
    stereo_grid.update(stereo)
    tof_grid.update(tof)
    overlap = (stereo_grid.data == OCCUPIED) & (tof_grid.data == OCCUPIED)
    assert np.count_nonzero(overlap) >= 8
    # Shared center obstacle must be near y=3, not Euclidean range projected as y.
    rows = np.nonzero(overlap)[0]
    y = config.y_min_m + (rows + 0.5) * config.resolution_m
    assert np.any(np.abs(y - 3.0) < 0.2)


def test_dense_mock_stereo_shape_and_finite_filtering() -> None:
    scene = make_noisy_multisensor_scene(480, 640)
    depth = disparity_to_depth(scene.disparity_px, scene.calibration, max_depth_m=12)
    points = valid_points(camera_to_vehicle(backproject_depth(depth, scene.calibration)))
    assert depth.shape == (480, 640)
    assert 250_000 < len(points) < depth.size
    assert np.isfinite(points).all()
