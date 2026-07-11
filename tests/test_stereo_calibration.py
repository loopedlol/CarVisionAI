from pathlib import Path

import cv2
import numpy as np
import pytest

from carvision.frames import CAMERA_OPTICAL_TO_VEHICLE
from carvision.stereo_calibration import CheckerboardConfig, calibrate_stereo_folders


def test_checkerboard_config_validation() -> None:
    with pytest.raises(ValueError):
        CheckerboardConfig(2, 6, 0.02)
    with pytest.raises(ValueError):
        CheckerboardConfig(9, 6, 0)


def test_calibration_reports_insufficient_detectable_pairs(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir(); right.mkdir()
    blank = np.zeros((80, 100), np.uint8)
    for index in range(3):
        cv2.imwrite(str(left / f"{index}.png"), blank)
        cv2.imwrite(str(right / f"{index}.png"), blank)
    with pytest.raises(ValueError, match="usable checkerboard pairs"):
        calibrate_stereo_folders(
            left, right, CheckerboardConfig(9, 6, 0.025),
            camera_to_vehicle=CAMERA_OPTICAL_TO_VEHICLE, min_pairs=2)
