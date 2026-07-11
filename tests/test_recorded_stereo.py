from pathlib import Path

import cv2
import numpy as np
import pytest

from carvision.frames import CAMERA_OPTICAL_TO_VEHICLE, RigidTransform
from carvision.mapping import OccupancyGridConfig
from carvision.recorded_stereo import (
    RecordedStereoAdapter,
    SGBMConfig,
    StereoRigCalibration,
    _left_right_consistency,
    matched_image_pairs,
    save_diagnostics,
)


def calibration(width: int = 192, height: int = 96,
                transform: RigidTransform = CAMERA_OPTICAL_TO_VEHICLE) -> StereoRigCalibration:
    focal = 140.0
    matrix = np.array([[focal, 0, (width - 1) / 2],
                       [0, focal, (height - 1) / 2], [0, 0, 1.0]])
    return StereoRigCalibration(
        (width, height), matrix, np.zeros(5), matrix.copy(), np.zeros(5),
        np.eye(3), np.array([-0.12, 0.0, 0.0]), transform, 0.2,
    )


def shifted_pair(width: int = 192, height: int = 96, disparity: int = 8
                 ) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    left = rng.integers(0, 256, (height, width), dtype=np.uint8)
    left = cv2.GaussianBlur(left, (3, 3), 0)
    right = np.zeros_like(left)
    right[:, :-disparity] = left[:, disparity:]
    right[:, -disparity:] = rng.integers(0, 256, (height, disparity), dtype=np.uint8)
    return left, right


def test_calibration_json_round_trip(tmp_path: Path) -> None:
    original = calibration(transform=RigidTransform(
        CAMERA_OPTICAL_TO_VEHICLE.rotation, np.array([0.1, 0.3, 1.0])))
    path = tmp_path / "rig.json"
    original.save(path)
    loaded = StereoRigCalibration.load(path)
    assert loaded.image_size == (192, 96)
    np.testing.assert_allclose(loaded.right_from_left_translation_m, [-0.12, 0, 0])
    np.testing.assert_allclose(loaded.camera_to_vehicle.translation_m, [0.1, 0.3, 1.0])
    assert loaded.to_dict()["units"]["translation"] == "metres"


def test_calibration_rejects_bad_baseline_and_version() -> None:
    with pytest.raises(ValueError):
        StereoRigCalibration(
            (20, 10), np.eye(3), np.zeros(5), np.eye(3), np.zeros(5),
            np.eye(3), np.zeros(3))
    data = calibration().to_dict()
    data["format_version"] = 99
    with pytest.raises(ValueError):
        StereoRigCalibration.from_dict(data)


def test_rectification_validates_dimensions_and_is_cached() -> None:
    adapter = RecordedStereoAdapter(calibration())
    left, right = shifted_pair()
    map_ids = tuple(id(item) for item in adapter._maps or ())
    rectified = adapter.rectify(left, right)
    assert rectified[0].shape == left.shape
    assert tuple(id(item) for item in adapter._maps or ()) == map_ids
    with pytest.raises(ValueError, match="calibration requires"):
        adapter.rectify(left[:, :-1], right[:, :-1])
    with pytest.raises(ValueError, match="dimensions differ"):
        adapter.rectify(left, right[:, :-1])


def test_actual_sgbm_shift_direction_and_fixed_point_conversion() -> None:
    left, right = shifted_pair(disparity=8)
    adapter = RecordedStereoAdapter(
        calibration(), SGBMConfig(num_disparities=64, block_size=5,
                                  speckle_window_size=0, left_right_max_diff_px=1.5,
                                  min_depth_m=0.5, max_depth_m=10.0))
    rect_left, rect_right = adapter.rectify(left, right)
    disparity, valid = adapter.compute_disparity(rect_left, rect_right)
    interior = valid[:, 80:-16]
    assert np.count_nonzero(interior) > 1000
    assert np.median(disparity[:, 80:-16][interior]) == pytest.approx(8.0, abs=0.3)


def test_positive_min_disparity_value_itself_is_not_treated_as_invalid() -> None:
    left, right = shifted_pair(disparity=8)
    adapter = RecordedStereoAdapter(
        calibration(), SGBMConfig(min_disparity=8, num_disparities=64,
                                  speckle_window_size=0, left_right_max_diff_px=None,
                                  min_depth_m=0.5, max_depth_m=10))
    disparity, valid = adapter.compute_disparity(*adapter.rectify(left, right))
    selected = valid[:, 80:-16]
    assert np.count_nonzero(selected) > 1000
    assert np.median(disparity[:, 80:-16][selected]) == pytest.approx(8.0, abs=0.3)


def test_left_right_consistency_rejects_occlusion_and_disagreement() -> None:
    left = np.full((2, 8), 2.0, np.float32)
    right = np.full((2, 8), -2.0, np.float32)
    right[:, 2] = -5.0
    mask = _left_right_consistency(left, right, 0.5)
    assert not mask[:, :2].any()  # x - disparity lies outside
    assert not mask[:, 4].any()   # samples disagreeing right pixel 2
    assert mask[:, 5:].all()


def test_pipeline_uses_existing_geometry_extrinsic_and_sensor_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    transform = RigidTransform(CAMERA_OPTICAL_TO_VEHICLE.rotation, np.array([0.2, 0.5, 1.0]))
    adapter = RecordedStereoAdapter(
        calibration(transform=transform),
        SGBMConfig(num_disparities=64, min_depth_m=0.5, max_depth_m=10,
                   left_right_max_diff_px=None),
        OccupancyGridConfig(-5, 5, -1, 9, 0.1, -0.2, 1.5),
    )
    expected_disparity = adapter.rectified_geometry.fx_px * 0.12 / 3.0

    def known_disparity(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(left.shape, expected_disparity, np.float32)
        valid = np.zeros(left.shape, bool)
        valid[left.shape[0] // 2, left.shape[1] // 2] = True
        return values, valid

    monkeypatch.setattr(adapter, "compute_disparity", known_disparity)
    left, right = shifted_pair()
    result = adapter.process_arrays(left, right)
    assert len(result.points_vehicle_m) == 1
    point = result.points_vehicle_m[0]
    assert point[1] == pytest.approx(3.5, abs=0.03)
    assert point[2] == pytest.approx(1.0, abs=0.03)
    assert np.count_nonzero(result.occupancy.data == 100) == 1


def test_folder_matching_and_missing_counterpart(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    image = np.zeros((5, 5), np.uint8)
    cv2.imwrite(str(left / "0001.png"), image)
    cv2.imwrite(str(right / "0001.png"), image)
    assert len(matched_image_pairs(left, right)) == 1
    cv2.imwrite(str(left / "0002.png"), image)
    with pytest.raises(ValueError, match="missing right"):
        matched_image_pairs(left, right)


def test_saved_diagnostics_are_real_and_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = RecordedStereoAdapter(
        calibration(), SGBMConfig(num_disparities=64, left_right_max_diff_px=None,
                                  min_depth_m=0.5, max_depth_m=10))
    disparity = adapter.rectified_geometry.fx_px * 0.12 / 3.0
    monkeypatch.setattr(adapter, "compute_disparity", lambda left, right: (
        np.full(left.shape, disparity, np.float32), np.ones(left.shape, bool)))
    left, right = shifted_pair()
    result = adapter.process_arrays(left, right)
    save_diagnostics(tmp_path, left, right, result)
    expected = {"original_pair.png", "rectified_pair_guides.png", "disparity.png",
                "validity_mask.png", "depth_m.png", "occupancy.png", "summary.json",
                "filtered_disparity_px.npy", "depth_m.npy", "points_vehicle_m.npy",
                "occupancy.npy"}
    assert expected <= {path.name for path in tmp_path.iterdir()}
    assert cv2.imread(str(tmp_path / "rectified_pair_guides.png")).shape == (96, 384, 3)
