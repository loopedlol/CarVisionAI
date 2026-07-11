"""Checkerboard-based calibration for recorded stereo folders."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .frames import RigidTransform
from .recorded_stereo import StereoRigCalibration, matched_image_pairs


@dataclass(frozen=True)
class CheckerboardConfig:
    columns: int
    rows: int
    square_size_m: float

    def __post_init__(self) -> None:
        if self.columns < 3 or self.rows < 3 or self.square_size_m <= 0:
            raise ValueError("checkerboard requires >=3x3 inner corners and positive square size")


@dataclass(frozen=True)
class CalibrationReport:
    calibration: StereoRigCalibration
    used_pairs: int
    rejected_pairs: int
    left_rms_px: float
    right_rms_px: float
    stereo_rms_px: float


def calibrate_stereo_folders(
    left_dir: Path | str,
    right_dir: Path | str,
    checkerboard: CheckerboardConfig,
    *,
    camera_to_vehicle: RigidTransform,
    min_pairs: int = 8,
) -> CalibrationReport:
    """Detect synchronized checkerboards, calibrate cameras, then stereo extrinsics."""
    pairs = matched_image_pairs(left_dir, right_dir)
    object_template = np.zeros((checkerboard.rows * checkerboard.columns, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:checkerboard.columns, 0:checkerboard.rows].T.reshape(-1, 2)
    object_template *= checkerboard.square_size_m
    objects: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    rejected = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-5)
    for left_path, right_path in pairs:
        left = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            rejected += 1
            continue
        if left.shape != right.shape:
            raise ValueError(f"calibration pair dimensions differ: {left_path}, {right_path}")
        current_size = (left.shape[1], left.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError("all calibration images must have identical dimensions")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(left, (checkerboard.columns, checkerboard.rows), flags)
        found_r, corners_r = cv2.findChessboardCorners(right, (checkerboard.columns, checkerboard.rows), flags)
        if not found_l or not found_r:
            rejected += 1
            continue
        left_points.append(cv2.cornerSubPix(left, corners_l, (11, 11), (-1, -1), criteria))
        right_points.append(cv2.cornerSubPix(right, corners_r, (11, 11), (-1, -1), criteria))
        objects.append(object_template.copy())
    if image_size is None or len(objects) < min_pairs:
        raise ValueError(f"only {len(objects)} usable checkerboard pairs; require at least {min_pairs}")

    left_rms, left_k, left_d, _, _ = cv2.calibrateCamera(objects, left_points, image_size, None, None)
    right_rms, right_k, right_d, _, _ = cv2.calibrateCamera(objects, right_points, image_size, None, None)
    stereo_rms, left_k, left_d, right_k, right_d, rotation, translation, _, _ = cv2.stereoCalibrate(
        objects, left_points, right_points, left_k, left_d, right_k, right_d, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    calibration = StereoRigCalibration(
        image_size, left_k, left_d, right_k, right_d, rotation, translation.reshape(3),
        camera_to_vehicle, float(stereo_rms),
    )
    return CalibrationReport(calibration, len(objects), rejected, float(left_rms),
                             float(right_rms), float(stereo_rms))


def draw_detected_checkerboard(image_path: Path | str, checkerboard: CheckerboardConfig) -> np.ndarray:
    """Return a diagnostic image with detected inner corners, or raise."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (checkerboard.columns, checkerboard.rows),
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise ValueError("checkerboard not detected")
    cv2.drawChessboardCorners(image, (checkerboard.columns, checkerboard.rows), corners, found)
    return image
