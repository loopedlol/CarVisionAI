import numpy as np
import pytest

from carvision.frames import camera_to_vehicle
from carvision.stereo import StereoCalibration, backproject_depth, disparity_to_depth


CAL = StereoCalibration(100.0, 100.0, 1.0, 1.0, 0.2)


def test_disparity_to_metric_depth_and_invalids() -> None:
    disparity = np.array([[10.0, 0.0, -1.0, np.nan]])
    depth = disparity_to_depth(disparity, CAL)
    assert depth[0, 0] == pytest.approx(2.0)
    assert np.isnan(depth[0, 1:]).all()


def test_backprojection_at_principal_point_and_frame_conversion() -> None:
    depth = np.full((3, 3), np.nan)
    depth[1, 1] = 2.0
    depth[0, 2] = 4.0
    camera = backproject_depth(depth, CAL)
    np.testing.assert_allclose(camera[1, 1], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(camera[0, 2], [0.04, -0.04, 4.0])
    np.testing.assert_allclose(camera_to_vehicle(camera)[0, 2], [0.04, 4.0, 0.04])


def test_backprojection_rejects_non_image() -> None:
    with pytest.raises(ValueError):
        backproject_depth(np.ones(3), CAL)

