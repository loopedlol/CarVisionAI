"""Recorded rectified-stereo adapter built on the core geometry and mapper."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .frames import CAMERA_OPTICAL_TO_VEHICLE, RigidTransform
from .mapping import OccupancyGrid, OccupancyGridConfig
from .stereo import StereoCalibration, backproject_depth, disparity_to_depth, valid_points


@dataclass(frozen=True)
class StereoRigCalibration:
    """OpenCV stereo calibration plus left-optical-camera vehicle extrinsics."""

    image_size: tuple[int, int]  # width, height
    left_camera_matrix: NDArray[np.float64]
    left_distortion: NDArray[np.float64]
    right_camera_matrix: NDArray[np.float64]
    right_distortion: NDArray[np.float64]
    right_from_left_rotation: NDArray[np.float64]
    right_from_left_translation_m: NDArray[np.float64]
    camera_to_vehicle: RigidTransform = CAMERA_OPTICAL_TO_VEHICLE
    calibration_rms_px: float | None = None

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must be positive (width, height)")
        matrices = {
            "left_camera_matrix": (self.left_camera_matrix, (3, 3)),
            "right_camera_matrix": (self.right_camera_matrix, (3, 3)),
            "right_from_left_rotation": (self.right_from_left_rotation, (3, 3)),
            "right_from_left_translation_m": (self.right_from_left_translation_m, (3,)),
        }
        for name, (value, shape) in matrices.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, array.copy())
        for name in ("left_distortion", "right_distortion"):
            array = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if array.size not in (4, 5, 8, 12, 14) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain a supported OpenCV distortion vector")
            object.__setattr__(self, name, array.copy())
        if not np.allclose(self.right_from_left_rotation.T @ self.right_from_left_rotation,
                           np.eye(3), atol=1e-6):
            raise ValueError("right_from_left_rotation must be orthonormal")
        baseline = np.linalg.norm(self.right_from_left_translation_m)
        if not np.isfinite(baseline) or baseline <= 1e-6:
            raise ValueError("stereo translation must provide a non-zero baseline in metres")
        for matrix_name in ("left_camera_matrix", "right_camera_matrix"):
            matrix = getattr(self, matrix_name)
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.isclose(matrix[2, 2], 1.0):
                raise ValueError(f"{matrix_name} has invalid pinhole intrinsics")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "units": {"translation": "metres", "pixels": "pixels"},
            "image_size": list(self.image_size),
            "left": {"camera_matrix": self.left_camera_matrix.tolist(),
                     "distortion": self.left_distortion.tolist()},
            "right": {"camera_matrix": self.right_camera_matrix.tolist(),
                      "distortion": self.right_distortion.tolist()},
            "right_from_left": {"rotation": self.right_from_left_rotation.tolist(),
                                "translation_m": self.right_from_left_translation_m.tolist()},
            "left_camera_to_vehicle": {"rotation": self.camera_to_vehicle.rotation.tolist(),
                                       "translation_m": self.camera_to_vehicle.translation_m.tolist()},
            "calibration_rms_px": self.calibration_rms_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StereoRigCalibration":
        if data.get("format_version") != 1:
            raise ValueError("unsupported or missing calibration format_version")
        try:
            return cls(
                tuple(data["image_size"]),
                np.asarray(data["left"]["camera_matrix"]),
                np.asarray(data["left"]["distortion"]),
                np.asarray(data["right"]["camera_matrix"]),
                np.asarray(data["right"]["distortion"]),
                np.asarray(data["right_from_left"]["rotation"]),
                np.asarray(data["right_from_left"]["translation_m"]),
                RigidTransform(np.asarray(data["left_camera_to_vehicle"]["rotation"]),
                               np.asarray(data["left_camera_to_vehicle"]["translation_m"])),
                data.get("calibration_rms_px"),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid calibration structure: missing {exc}") from exc

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "StereoRigCalibration":
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load calibration {path}: {exc}") from exc
        return cls.from_dict(data)


@dataclass(frozen=True)
class SGBMConfig:
    min_disparity: int = 0
    num_disparities: int = 128
    block_size: int = 5
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1
    pre_filter_cap: int = 31
    left_right_max_diff_px: float | None = 1.5
    min_depth_m: float = 0.3
    max_depth_m: float = 20.0

    def __post_init__(self) -> None:
        if self.num_disparities <= 0 or self.num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be odd and at least 3")
        if self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("depth limits must satisfy 0 < min < max")
        if self.left_right_max_diff_px is not None and self.left_right_max_diff_px < 0:
            raise ValueError("left_right_max_diff_px must be non-negative or None")


@dataclass
class RecordedStereoResult:
    rectified_left: NDArray[np.uint8]
    rectified_right: NDArray[np.uint8]
    raw_disparity_px: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    filtered_disparity_px: NDArray[np.float32]
    depth_m: NDArray[np.float64]
    points_vehicle_m: NDArray[np.float64]
    occupancy: OccupancyGrid
    timings_ms: dict[str, float]


class RecordedStereoAdapter:
    """Rectify, match, filter, back-project, transform, and map stereo pairs."""

    def __init__(self, calibration: StereoRigCalibration, sgbm: SGBMConfig = SGBMConfig(),
                 grid: OccupancyGridConfig = OccupancyGridConfig(y_min_m=-1.0)) -> None:
        self.rig = calibration
        self.sgbm = sgbm
        self.grid_config = grid
        self._maps: tuple[NDArray[np.float32], ...] | None = None
        self._rectified_geometry: StereoCalibration | None = None
        self._left_roi: tuple[int, int, int, int] | None = None
        self._right_roi: tuple[int, int, int, int] | None = None
        self._build_rectification_maps()

    @property
    def rectified_geometry(self) -> StereoCalibration:
        assert self._rectified_geometry is not None
        return self._rectified_geometry

    def _build_rectification_maps(self) -> None:
        size = self.rig.image_size
        r1, r2, p1, p2, _, roi1, roi2 = cv2.stereoRectify(
            self.rig.left_camera_matrix, self.rig.left_distortion,
            self.rig.right_camera_matrix, self.rig.right_distortion,
            size, self.rig.right_from_left_rotation,
            self.rig.right_from_left_translation_m.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        left_maps = cv2.initUndistortRectifyMap(
            self.rig.left_camera_matrix, self.rig.left_distortion, r1, p1, size, cv2.CV_32FC1)
        right_maps = cv2.initUndistortRectifyMap(
            self.rig.right_camera_matrix, self.rig.right_distortion, r2, p2, size, cv2.CV_32FC1)
        self._maps = (*left_maps, *right_maps)
        baseline = abs(float(p2[0, 3] / p2[0, 0]))
        self._rectified_geometry = StereoCalibration(float(p1[0, 0]), float(p1[1, 1]),
                                                     float(p1[0, 2]), float(p1[1, 2]), baseline)
        self._left_roi, self._right_roi = tuple(roi1), tuple(roi2)

    def validate_pair(self, left: NDArray[np.generic], right: NDArray[np.generic]) -> None:
        expected = (self.rig.image_size[1], self.rig.image_size[0])
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"left/right image dimensions differ: {left.shape[:2]} vs {right.shape[:2]}")
        if left.shape[:2] != expected:
            raise ValueError(f"images are {left.shape[:2][::-1]}, calibration requires {self.rig.image_size}")
        if left.ndim not in (2, 3) or right.ndim not in (2, 3):
            raise ValueError("images must be grayscale or color")

    def rectify(self, left: NDArray[np.generic], right: NDArray[np.generic]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        self.validate_pair(left, right)
        left_gray = _as_gray_u8(left)
        right_gray = _as_gray_u8(right)
        assert self._maps is not None
        return (cv2.remap(left_gray, self._maps[0], self._maps[1], cv2.INTER_LINEAR),
                cv2.remap(right_gray, self._maps[2], self._maps[3], cv2.INTER_LINEAR))

    def compute_disparity(self, left: NDArray[np.uint8], right: NDArray[np.uint8]
                          ) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        matcher = _create_matcher(self.sgbm)
        raw = matcher.compute(left, right).astype(np.float32) / 16.0
        # OpenCV uses (minDisparity - 1) as its invalid sentinel. Equality with
        # a positive minDisparity is a valid match; zero/negative disparities
        # remain incompatible with the standard positive-baseline depth model.
        valid = (np.isfinite(raw) & (raw > 0.0)
                 & (raw >= float(self.sgbm.min_disparity)))
        if self.sgbm.left_right_max_diff_px is not None:
            right_matcher = _create_right_matcher(self.sgbm)
            right_disparity = right_matcher.compute(right, left).astype(np.float32) / 16.0
            valid &= _left_right_consistency(raw, right_disparity, self.sgbm.left_right_max_diff_px)
        geometry = self.rectified_geometry
        minimum_d = geometry.fx_px * geometry.baseline_m / self.sgbm.max_depth_m
        maximum_d = geometry.fx_px * geometry.baseline_m / self.sgbm.min_depth_m
        valid &= (raw >= minimum_d) & (raw <= maximum_d)
        valid &= _roi_mask(raw.shape, self._left_roi) & _roi_mask(raw.shape, self._right_roi)
        return raw, valid

    def process_arrays(self, left: NDArray[np.generic], right: NDArray[np.generic]) -> RecordedStereoResult:
        timings: dict[str, float] = {}
        start = perf_counter()
        rect_left, rect_right = self.rectify(left, right)
        timings["rectification"] = (perf_counter() - start) * 1000
        start = perf_counter()
        raw, valid = self.compute_disparity(rect_left, rect_right)
        timings["disparity_and_filtering"] = (perf_counter() - start) * 1000
        filtered = np.where(valid, raw, np.nan).astype(np.float32)
        start = perf_counter()
        depth = disparity_to_depth(filtered, self.rectified_geometry,
                                   min_depth_m=self.sgbm.min_depth_m,
                                   max_depth_m=self.sgbm.max_depth_m)
        camera_cloud = backproject_depth(depth, self.rectified_geometry)
        vehicle_points = valid_points(self.rig.camera_to_vehicle.apply(camera_cloud))
        timings["geometry"] = (perf_counter() - start) * 1000
        start = perf_counter()
        occupancy = OccupancyGrid(self.grid_config)
        occupancy.update(vehicle_points, self.rig.camera_to_vehicle.translation_m)
        timings["mapping"] = (perf_counter() - start) * 1000
        timings["total"] = sum(timings.values())
        return RecordedStereoResult(rect_left, rect_right, raw, valid, filtered, depth,
                                    vehicle_points, occupancy, timings)

    def process_files(self, left_path: Path | str, right_path: Path | str) -> tuple[NDArray[np.uint8], NDArray[np.uint8], RecordedStereoResult]:
        start = perf_counter()
        left = _load_image(left_path)
        right = _load_image(right_path)
        loading_ms = (perf_counter() - start) * 1000
        result = self.process_arrays(left, right)
        result.timings_ms["image_loading"] = loading_ms
        result.timings_ms["total"] += loading_ms
        return left, right, result


def matched_image_pairs(left_dir: Path | str, right_dir: Path | str) -> list[tuple[Path, Path]]:
    """Match supported image files by identical relative path, rejecting gaps."""
    left_root, right_root = Path(left_dir), Path(right_dir)
    if not left_root.is_dir() or not right_root.is_dir():
        raise ValueError("left and right paths must both be directories")
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    left = {p.relative_to(left_root): p for p in left_root.rglob("*") if p.suffix.lower() in suffixes}
    right = {p.relative_to(right_root): p for p in right_root.rglob("*") if p.suffix.lower() in suffixes}
    missing_right, missing_left = sorted(left.keys() - right.keys()), sorted(right.keys() - left.keys())
    if missing_right or missing_left:
        raise ValueError(f"unmatched stereo files; missing right={missing_right[:5]}, missing left={missing_left[:5]}")
    if not left:
        raise ValueError("no supported images found")
    return [(left[name], right[name]) for name in sorted(left)]


def save_diagnostics(output_dir: Path | str, original_left: NDArray[np.uint8],
                     original_right: NDArray[np.uint8], result: RecordedStereoResult) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "original_pair.png"), np.hstack((_as_bgr(original_left), _as_bgr(original_right))))
    guides = np.hstack((_as_bgr(result.rectified_left), _as_bgr(result.rectified_right)))
    for y in range(20, guides.shape[0], 40):
        cv2.line(guides, (0, y), (guides.shape[1] - 1, y), (0, 255, 0), 1)
    cv2.imwrite(str(output / "rectified_pair_guides.png"), guides)
    cv2.imwrite(str(output / "disparity.png"), _scalar_preview(result.raw_disparity_px, result.valid_mask, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(output / "validity_mask.png"), result.valid_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "depth_m.png"), _scalar_preview(result.depth_m, np.isfinite(result.depth_m), cv2.COLORMAP_TURBO, invert=True))
    cv2.imwrite(str(output / "occupancy.png"), result.occupancy.preview_u8())
    np.save(output / "filtered_disparity_px.npy", result.filtered_disparity_px)
    np.save(output / "depth_m.npy", result.depth_m)
    np.save(output / "points_vehicle_m.npy", result.points_vehicle_m)
    np.save(output / "occupancy.npy", result.occupancy.data)
    summary = {
        "valid_pixels": int(np.count_nonzero(result.valid_mask)),
        "total_pixels": int(result.valid_mask.size),
        "valid_fraction": float(np.mean(result.valid_mask)),
        "depth_min_m": float(np.nanmin(result.depth_m)) if np.isfinite(result.depth_m).any() else None,
        "depth_max_m": float(np.nanmax(result.depth_m)) if np.isfinite(result.depth_m).any() else None,
        "point_count": len(result.points_vehicle_m),
        "point_min_vehicle_m": np.min(result.points_vehicle_m, axis=0).tolist() if len(result.points_vehicle_m) else None,
        "point_max_vehicle_m": np.max(result.points_vehicle_m, axis=0).tolist() if len(result.points_vehicle_m) else None,
        "timings_ms": result.timings_ms,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _create_matcher(config: SGBMConfig) -> cv2.StereoSGBM:
    channels = 1
    return cv2.StereoSGBM_create(
        minDisparity=config.min_disparity, numDisparities=config.num_disparities,
        blockSize=config.block_size, P1=8 * channels * config.block_size ** 2,
        P2=32 * channels * config.block_size ** 2,
        disp12MaxDiff=config.disp12_max_diff, preFilterCap=config.pre_filter_cap,
        uniquenessRatio=config.uniqueness_ratio, speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def _create_right_matcher(config: SGBMConfig) -> cv2.StereoSGBM:
    right_min = -config.min_disparity - config.num_disparities
    right_config = SGBMConfig(
        min_disparity=right_min, num_disparities=config.num_disparities,
        block_size=config.block_size, uniqueness_ratio=config.uniqueness_ratio,
        speckle_window_size=config.speckle_window_size, speckle_range=config.speckle_range,
        disp12_max_diff=config.disp12_max_diff, pre_filter_cap=config.pre_filter_cap,
        left_right_max_diff_px=None, min_depth_m=config.min_depth_m, max_depth_m=config.max_depth_m,
    )
    return _create_matcher(right_config)


def _left_right_consistency(left: NDArray[np.float32], right: NDArray[np.float32],
                            threshold_px: float) -> NDArray[np.bool_]:
    height, width = left.shape
    x = np.broadcast_to(np.arange(width), (height, width))
    right_x = np.rint(x - left).astype(np.int32)
    in_bounds = (right_x >= 0) & (right_x < width) & np.isfinite(left)
    sampled = np.full(left.shape, np.nan, dtype=np.float32)
    rows, cols = np.nonzero(in_bounds)
    sampled[rows, cols] = right[rows, right_x[rows, cols]]
    return in_bounds & np.isfinite(sampled) & (np.abs(left + sampled) <= threshold_px)


def _roi_mask(shape: tuple[int, int], roi: tuple[int, int, int, int] | None) -> NDArray[np.bool_]:
    mask = np.zeros(shape, dtype=bool)
    if roi is None:
        mask[:] = True
    else:
        x, y, width, height = roi
        mask[y:y + height, x:x + width] = True
    return mask


def _as_gray_u8(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    if image.dtype != np.uint8:
        raise ValueError("recorded images must be uint8")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(image, code)
    raise ValueError("images must be grayscale, BGR, or BGRA")


def _as_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[..., :3].copy()


def _load_image(path: Path | str) -> NDArray[np.uint8]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def _scalar_preview(values: NDArray[np.floating], valid: NDArray[np.bool_], colormap: int,
                    *, invert: bool = False) -> NDArray[np.uint8]:
    output = np.zeros(values.shape, dtype=np.uint8)
    finite = valid & np.isfinite(values)
    if np.any(finite):
        low, high = np.percentile(values[finite], [2, 98])
        if high <= low:
            high = low + 1.0
        normalized = np.clip((values[finite] - low) / (high - low), 0, 1)
        if invert:
            normalized = 1 - normalized
        output[finite] = np.rint(normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(output, colormap)
    colored[~finite] = 0
    return colored
