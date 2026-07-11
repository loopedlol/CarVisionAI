"""Objective evaluation tools for annotated recorded stereo datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from .mapping import OCCUPIED, OccupancyGridConfig
from .recorded_stereo import (
    RecordedStereoAdapter,
    RecordedStereoResult,
    SGBMConfig,
    StereoRigCalibration,
    save_diagnostics,
)


@dataclass(frozen=True)
class EvaluationRegion:
    name: str
    rect: tuple[int, int, int, int]  # x, y, width, height in rectified left image
    expected_depth_m: float | None = None
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        x, y, width, height = self.rect
        if not self.name or x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI needs a name and non-negative x/y with positive width/height")
        if self.expected_depth_m is not None and self.expected_depth_m <= 0:
            raise ValueError("expected_depth_m must be positive")


@dataclass(frozen=True)
class EvaluationFrame:
    frame_id: str
    left_path: Path
    right_path: Path
    regions: tuple[EvaluationRegion, ...] = ()


@dataclass(frozen=True)
class EvaluationScene:
    scene_id: str
    category: str
    static: bool
    frames: tuple[EvaluationFrame, ...]
    notes: str = ""
    conditions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationDataset:
    root: Path
    name: str
    calibration_path: Path
    expected_image_size: tuple[int, int]
    scenes: tuple[EvaluationScene, ...]
    default_sgbm: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def load(cls, manifest_path: Path | str) -> "EvaluationDataset":
        manifest = Path(manifest_path).resolve()
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load dataset manifest: {exc}") from exc
        if data.get("format_version") != 1:
            raise ValueError("unsupported or missing dataset format_version")
        root = manifest.parent
        try:
            size = tuple(int(value) for value in data["expected_image_size"])
            if len(size) != 2 or min(size) <= 0:
                raise ValueError
            scenes: list[EvaluationScene] = []
            scene_ids: set[str] = set()
            for raw_scene in data["scenes"]:
                scene_id = str(raw_scene["id"])
                if not scene_id or scene_id in scene_ids:
                    raise ValueError(f"duplicate or empty scene id: {scene_id}")
                scene_ids.add(scene_id)
                frames: list[EvaluationFrame] = []
                frame_ids: set[str] = set()
                for raw_frame in raw_scene["frames"]:
                    frame_id = str(raw_frame["id"])
                    if not frame_id or frame_id in frame_ids:
                        raise ValueError(f"duplicate or empty frame id in {scene_id}: {frame_id}")
                    frame_ids.add(frame_id)
                    regions = tuple(EvaluationRegion(
                        str(region["name"]), tuple(int(v) for v in region["rect"]),
                        region.get("expected_depth_m"), tuple(region.get("tags", ())),
                        str(region.get("notes", "")),
                    ) for region in raw_frame.get("regions", ()))
                    frames.append(EvaluationFrame(
                        frame_id, _resolve_dataset_path(root, raw_frame["left"]),
                        _resolve_dataset_path(root, raw_frame["right"]), regions))
                if not frames:
                    raise ValueError(f"scene {scene_id} has no frames")
                scenes.append(EvaluationScene(
                    scene_id, str(raw_scene.get("category", "unspecified")),
                    bool(raw_scene.get("static", False)), tuple(frames),
                    str(raw_scene.get("notes", "")),
                    {str(k): str(v) for k, v in raw_scene.get("conditions", {}).items()},
                ))
            if not scenes:
                raise ValueError("dataset has no scenes")
            calibration = _resolve_dataset_path(root, data["calibration"])
            dataset = cls(root, str(data["name"]), calibration, size, tuple(scenes),
                          dict(data.get("default_sgbm", {})), str(data.get("notes", "")))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid dataset structure: {exc}") from exc
        if not dataset.calibration_path.is_file():
            raise ValueError(f"calibration file does not exist: {dataset.calibration_path}")
        calibration_data = StereoRigCalibration.load(dataset.calibration_path)
        if calibration_data.image_size != dataset.expected_image_size:
            raise ValueError("dataset expected_image_size does not match calibration")
        for frame in dataset.frames():
            if not frame.left_path.is_file() or not frame.right_path.is_file():
                raise ValueError(f"missing image for frame {frame.frame_id}")
            for region in frame.regions:
                x, y, width, height = region.rect
                if x + width > size[0] or y + height > size[1]:
                    raise ValueError(f"ROI {region.name} exceeds expected image dimensions")
        return dataset

    def frames(self) -> Iterable[EvaluationFrame]:
        for scene in self.scenes:
            yield from scene.frames

    def find(self, scene_id: str, frame_id: str) -> tuple[EvaluationScene, EvaluationFrame]:
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                for frame in scene.frames:
                    if frame.frame_id == frame_id:
                        return scene, frame
        raise ValueError(f"unknown scene/frame: {scene_id}/{frame_id}")


def sgbm_config_from_dict(values: dict[str, Any], base: SGBMConfig = SGBMConfig()) -> SGBMConfig:
    allowed = set(asdict(base))
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown SGBM parameters: {sorted(unknown)}")
    return replace(base, **values)


def vertical_rectification_metrics(
    left_rectified: NDArray[np.uint8], right_rectified: NDArray[np.uint8], *,
    max_features: int = 1500, ratio_threshold: float = 0.75,
) -> dict[str, float | int | None]:
    """Estimate vertical epipolar residual from ratio-tested ORB correspondences."""
    orb = cv2.ORB_create(nfeatures=max_features)
    left_keypoints, left_desc = orb.detectAndCompute(left_rectified, None)
    right_keypoints, right_desc = orb.detectAndCompute(right_rectified, None)
    if left_desc is None or right_desc is None or len(right_desc) < 2:
        return {"feature_match_count": 0, "vertical_error_median_px": None,
                "vertical_error_p95_px": None, "vertical_error_max_px": None}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_desc, right_desc, k=2)
    errors = []
    for nearest in pairs:
        if len(nearest) != 2 or nearest[0].distance >= ratio_threshold * nearest[1].distance:
            continue
        left_point = left_keypoints[nearest[0].queryIdx].pt
        right_point = right_keypoints[nearest[0].trainIdx].pt
        # Standard positive-disparity rigs require the right match not to lie
        # substantially to the right. This rejects many unrelated descriptors.
        if left_point[0] + 1.0 < right_point[0]:
            continue
        errors.append(abs(left_point[1] - right_point[1]))
    if not errors:
        return {"feature_match_count": 0, "vertical_error_median_px": None,
                "vertical_error_p95_px": None, "vertical_error_max_px": None}
    values = np.asarray(errors)
    return {"feature_match_count": len(errors),
            "vertical_error_median_px": float(np.median(values)),
            "vertical_error_p95_px": float(np.percentile(values, 95)),
            "vertical_error_max_px": float(np.max(values))}


def validity_metrics(valid_mask: NDArray[np.bool_], border_width_px: int) -> dict[str, float | int]:
    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2 or border_width_px < 0:
        raise ValueError("valid_mask must be 2D and border width non-negative")
    width = min(border_width_px, mask.shape[1])
    border = np.zeros(mask.shape, dtype=bool)
    border[:, :width] = True
    invalid = ~mask
    invalid_count = int(np.count_nonzero(invalid))
    return {
        "valid_pixel_count": int(np.count_nonzero(mask)),
        "valid_disparity_fraction": float(np.mean(mask)),
        "invalid_disparity_fraction": float(np.mean(invalid)),
        "invalid_border_fraction": float(np.mean(invalid[border])) if np.any(border) else 0.0,
        "invalid_pixels_in_search_border_fraction": (
            float(np.count_nonzero(invalid & border) / invalid_count) if invalid_count else 0.0),
        "search_border_width_px": width,
    }


def region_depth_metrics(depth_m: NDArray[np.floating], region: EvaluationRegion) -> dict[str, Any]:
    x, y, width, height = region.rect
    values = np.asarray(depth_m, dtype=np.float64)[y:y + height, x:x + width]
    finite = values[np.isfinite(values) & (values > 0)]
    output: dict[str, Any] = {
        "name": region.name, "rect": list(region.rect), "tags": list(region.tags),
        "expected_depth_m": region.expected_depth_m,
        "valid_fraction": float(finite.size / values.size), "valid_count": int(finite.size),
        "depth_median_m": None, "depth_mad_m": None, "depth_std_m": None,
        "depth_p10_m": None, "depth_p90_m": None,
        "absolute_error_m": None, "percentage_error": None,
    }
    if finite.size:
        median = float(np.median(finite))
        output.update(depth_median_m=median,
                      depth_mad_m=float(np.median(np.abs(finite - median))),
                      depth_std_m=float(np.std(finite)),
                      depth_p10_m=float(np.percentile(finite, 10)),
                      depth_p90_m=float(np.percentile(finite, 90)))
        if region.expected_depth_m is not None:
            error = abs(median - region.expected_depth_m)
            output["absolute_error_m"] = float(error)
            output["percentage_error"] = float(100.0 * error / region.expected_depth_m)
    return output


def occupancy_metrics(data: NDArray[np.integer], *, isolated_max_cells: int = 2) -> dict[str, int]:
    occupied = (np.asarray(data) == OCCUPIED).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(occupied, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=int)
    isolated_components = sizes <= isolated_max_cells
    return {
        "occupied_cell_count": int(np.count_nonzero(occupied)),
        "occupied_component_count": int(len(sizes)),
        "isolated_component_count": int(np.count_nonzero(isolated_components)),
        "isolated_occupied_cell_count": int(np.sum(sizes[isolated_components])),
    }


def occupancy_pair_stability(first: NDArray[np.integer], second: NDArray[np.integer]) -> dict[str, float | int]:
    a, b = np.asarray(first) == OCCUPIED, np.asarray(second) == OCCUPIED
    if a.shape != b.shape:
        raise ValueError("occupancy grids must have equal shapes")
    union = np.count_nonzero(a | b)
    intersection = np.count_nonzero(a & b)
    return {
        "occupied_iou": float(intersection / union) if union else 1.0,
        "occupied_intersection_cells": int(intersection),
        "occupied_union_cells": int(union),
        "occupied_changed_fraction": float(np.mean(a != b)),
    }


def temporal_region_metrics(frame_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[float]] = {}
    for frame in frame_metrics:
        for region in frame["regions"]:
            median = region["depth_median_m"]
            if median is not None:
                by_name.setdefault(region["name"], []).append(float(median))
    output = []
    for name, samples in sorted(by_name.items()):
        values = np.asarray(samples)
        median = float(np.median(values))
        output.append({"name": name, "sample_count": len(samples), "median_depth_m": median,
                       "temporal_std_m": float(np.std(values)),
                       "temporal_mad_m": float(np.median(np.abs(values - median))),
                       "temporal_range_m": float(np.ptp(values)),
                       "temporal_cv_percent": float(100 * np.std(values) / median) if median else None})
    return output


def evaluate_result(result: RecordedStereoResult, regions: tuple[EvaluationRegion, ...],
                    border_width_px: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    metrics.update(vertical_rectification_metrics(result.rectified_left, result.rectified_right))
    metrics.update(validity_metrics(result.valid_mask, border_width_px))
    metrics.update(occupancy_metrics(result.occupancy.data))
    metrics["regions"] = [region_depth_metrics(result.depth_m, region) for region in regions]
    metrics["timings_ms"] = dict(result.timings_ms)
    return metrics


def evaluate_frame(adapter: RecordedStereoAdapter, scene: EvaluationScene, frame: EvaluationFrame,
                   output_dir: Path | str | None = None) -> tuple[dict[str, Any], RecordedStereoResult]:
    left, right, result = adapter.process_files(frame.left_path, frame.right_path)
    metrics = evaluate_result(result, frame.regions,
                              adapter.sgbm.num_disparities + max(adapter.sgbm.min_disparity, 0))
    metrics.update(scene_id=scene.scene_id, frame_id=frame.frame_id,
                   category=scene.category, static_scene=scene.static,
                   conditions=scene.conditions)
    if output_dir is not None:
        output = Path(output_dir)
        save_diagnostics(output, left, right, result)
        save_failure_visualization(output / "annotated_regions.png", result, frame.regions)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, result


def evaluate_dataset(dataset: EvaluationDataset, config: SGBMConfig,
                     output_dir: Path | str, *, save_frame_diagnostics: bool = True) -> dict[str, Any]:
    calibration = StereoRigCalibration.load(dataset.calibration_path)
    adapter = RecordedStereoAdapter(calibration, config,
                                    OccupancyGridConfig(y_min_m=-1.0, y_max_m=19.0))
    output = Path(output_dir)
    frame_metrics: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    for scene in dataset.scenes:
        previous_occupancy: NDArray[np.integer] | None = None
        stability: list[dict[str, Any]] = []
        current_frames: list[dict[str, Any]] = []
        for frame in scene.frames:
            frame_output = output / "frames" / scene.scene_id / frame.frame_id if save_frame_diagnostics else None
            metrics, result = evaluate_frame(adapter, scene, frame, frame_output)
            frame_metrics.append(metrics)
            current_frames.append(metrics)
            if previous_occupancy is not None:
                pair = occupancy_pair_stability(previous_occupancy, result.occupancy.data)
                pair.update(previous_frame_id=scene.frames[len(current_frames) - 2].frame_id,
                            frame_id=frame.frame_id)
                stability.append(pair)
            previous_occupancy = result.occupancy.data.copy()
        scene_metrics.append({
            "scene_id": scene.scene_id, "category": scene.category, "static": scene.static,
            "frame_count": len(scene.frames), "conditions": scene.conditions,
            "temporal_regions": temporal_region_metrics(current_frames) if scene.static else [],
            "occupancy_stability": stability,
        })
    report = {
        "format_version": 1, "dataset": dataset.name, "configuration": asdict(config),
        "calibration_metrics": calibration_metrics(calibration),
        "frame_metrics": frame_metrics, "scene_metrics": scene_metrics,
        "summary": summarize_evaluation(frame_metrics, scene_metrics),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "summary.md").write_text(render_markdown_report(report))
    return report


def summarize_evaluation(frame_metrics: list[dict[str, Any]],
                         scene_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric(path: tuple[str, ...]) -> list[float]:
        values = []
        for frame in frame_metrics:
            value: Any = frame
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None and np.isfinite(value):
                values.append(float(value))
        return values
    valid = numeric(("valid_disparity_fraction",))
    invalid_border = numeric(("invalid_border_fraction",))
    rectification = numeric(("vertical_error_p95_px",))
    totals = numeric(("timings_ms", "total"))
    depth_errors = [float(region["percentage_error"])
                    for frame in frame_metrics for region in frame["regions"]
                    if region["percentage_error"] is not None]
    ious = [float(pair["occupied_iou"]) for scene in scene_metrics
            for pair in scene["occupancy_stability"]]
    isolated = numeric(("isolated_occupied_cell_count",))
    temporal_std = [float(region["temporal_std_m"]) for scene in scene_metrics
                    for region in scene.get("temporal_regions", [])]
    stages = sorted({stage for frame in frame_metrics for stage in frame["timings_ms"]})
    return {
        "frame_count": len(frame_metrics),
        "valid_disparity_fraction_median": _median_or_none(valid),
        "invalid_border_fraction_median": _median_or_none(invalid_border),
        "rectification_p95_px_median": _median_or_none(rectification),
        "depth_percentage_error_median": _median_or_none(depth_errors),
        "occupancy_iou_median": _median_or_none(ious),
        "isolated_occupied_cells_median": _median_or_none(isolated),
        "temporal_depth_std_m_max": max(temporal_std) if temporal_std else None,
        "total_runtime_ms_median": _median_or_none(totals),
        "total_runtime_ms_p95": _percentile_or_none(totals, 95),
        "runtime_ms_median_by_stage": {
            stage: _median_or_none([float(frame["timings_ms"][stage]) for frame in frame_metrics
                                    if stage in frame["timings_ms"]]) for stage in stages},
    }


def calibration_metrics(calibration: StereoRigCalibration) -> dict[str, Any]:
    """Expose stored fit quality and basic physical plausibility diagnostics."""
    translation = calibration.right_from_left_translation_m
    baseline = float(np.linalg.norm(translation))
    rotation_angle = float(np.rad2deg(np.arccos(np.clip(
        (np.trace(calibration.right_from_left_rotation) - 1) / 2, -1, 1))))
    return {
        "image_size": list(calibration.image_size),
        "calibration_rms_px": calibration.calibration_rms_px,
        "baseline_m": baseline,
        "right_from_left_translation_m": translation.tolist(),
        "relative_rotation_angle_deg": rotation_angle,
        "nonhorizontal_translation_fraction": float(np.linalg.norm(translation[1:]) / baseline),
        "left_focal_px": [float(calibration.left_camera_matrix[0, 0]),
                          float(calibration.left_camera_matrix[1, 1])],
    }


def compare_configurations(dataset: EvaluationDataset, named_configs: dict[str, SGBMConfig],
                           output_dir: Path | str) -> dict[str, Any]:
    if not 1 <= len(named_configs) <= 12:
        raise ValueError("comparison requires between 1 and 12 explicit configurations")
    output = Path(output_dir)
    comparisons = []
    for name, config in named_configs.items():
        if not name or "/" in name or ".." in name:
            raise ValueError(f"unsafe or empty configuration name: {name}")
        report = evaluate_dataset(dataset, config, output / name, save_frame_diagnostics=False)
        comparisons.append({"name": name, "configuration": asdict(config), **report["summary"]})
    result = {"format_version": 1, "dataset": dataset.name, "configurations": comparisons}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "comparison.md").write_text(render_comparison_markdown(result))
    return result


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    calibration = report.get("calibration_metrics", {})
    lines = [f"# Stereo evaluation: {report['dataset']}", "", "## Summary", "",
             "| Metric | Value |", "|---|---:|",
             f"| Frames | {summary['frame_count']} |",
             f"| Median valid disparity | {_format_percent_fraction(summary['valid_disparity_fraction_median'])} |",
             f"| Median invalid search border | {_format_percent_fraction(summary.get('invalid_border_fraction_median'))} |",
             f"| Median rectification p95 | {_format_number(summary['rectification_p95_px_median'], ' px')} |",
             f"| Median measured-depth error | {_format_number(summary['depth_percentage_error_median'], '%')} |",
             f"| Median consecutive occupancy IoU | {_format_number(summary['occupancy_iou_median'])} |",
             f"| Max temporal ROI depth std | {_format_number(summary.get('temporal_depth_std_m_max'), ' m')} |",
             f"| Median isolated occupied cells | {_format_number(summary.get('isolated_occupied_cells_median'))} |",
             f"| Median runtime | {_format_number(summary['total_runtime_ms_median'], ' ms')} |",
             "", "## Calibration", "",
             f"- Stored calibration RMS: {_format_number(calibration.get('calibration_rms_px'), ' px')}",
             f"- Baseline: {_format_number(calibration.get('baseline_m'), ' m')}",
             f"- Relative rotation angle: {_format_number(calibration.get('relative_rotation_angle_deg'), ' deg')}",
             f"- Non-horizontal translation fraction: {_format_percent_fraction(calibration.get('nonhorizontal_translation_fraction'))}",
             "", "## Flagged cases", ""]
    flags = _failure_flags(report)
    if flags:
        lines += [f"- {flag}" for flag in flags]
    else:
        lines.append("- None under default thresholds.")
    lines += ["", "Thresholds: rectification p95 > 1 px or <20 matches, depth error > 10%, frame validity < 20%, ROI validity < 30%, consecutive occupancy IoU < 0.5.", ""]
    return "\n".join(lines)


def render_comparison_markdown(result: dict[str, Any]) -> str:
    lines = [f"# Stereo parameter comparison: {result['dataset']}", "",
             "Configurations are evaluated on identical annotated frames.", "",
             "| Configuration | Valid | Depth error | Rect. p95 | Occupancy IoU | Runtime |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in result["configurations"]:
        lines.append(f"| {item['name']} | {_format_percent_fraction(item['valid_disparity_fraction_median'])} "
                     f"| {_format_number(item['depth_percentage_error_median'], '%')} "
                     f"| {_format_number(item['rectification_p95_px_median'], ' px')} "
                     f"| {_format_number(item['occupancy_iou_median'])} "
                     f"| {_format_number(item['total_runtime_ms_median'], ' ms')} |")
    lines += ["", "Interpret quality and runtime together; no single score is used.", ""]
    return "\n".join(lines)


def save_failure_visualization(path: Path | str, result: RecordedStereoResult,
                               regions: tuple[EvaluationRegion, ...]) -> None:
    image = cv2.cvtColor(result.rectified_left, cv2.COLOR_GRAY2BGR)
    for region in regions:
        x, y, width, height = region.rect
        valid_fraction = float(np.mean(result.valid_mask[y:y + height, x:x + width]))
        color = (0, 200, 0) if valid_fraction >= 0.5 else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), color, 2)
        label = f"{region.name} valid={valid_fraction:.0%}"
        cv2.putText(image, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _failure_flags(report: dict[str, Any]) -> list[str]:
    flags = []
    calibration = report.get("calibration_metrics", {})
    if calibration.get("calibration_rms_px") is not None and calibration["calibration_rms_px"] > 1.0:
        flags.append(f"calibration: stored RMS={calibration['calibration_rms_px']:.2f}px")
    if calibration.get("nonhorizontal_translation_fraction", 0) > 0.05:
        flags.append(f"calibration: non-horizontal translation={calibration['nonhorizontal_translation_fraction']:.1%} of baseline")
    for frame in report["frame_metrics"]:
        label = f"{frame['scene_id']}/{frame['frame_id']}"
        if frame["vertical_error_p95_px"] is not None and frame["vertical_error_p95_px"] > 1.0:
            flags.append(f"{label}: poor rectification p95={frame['vertical_error_p95_px']:.2f}px")
        if frame.get("feature_match_count", 0) < 20:
            flags.append(f"{label}: too few rectification feature matches={frame.get('feature_match_count', 0)}")
        if frame["valid_disparity_fraction"] < 0.2:
            flags.append(f"{label}: low disparity validity={frame['valid_disparity_fraction']:.1%}")
        for region in frame["regions"]:
            tags_value = region.get("tags", [])
            tags = f" tags={','.join(tags_value)}" if tags_value else ""
            if region.get("valid_fraction", 1.0) < 0.3:
                flags.append(f"{label}/{region['name']}: low ROI validity={region['valid_fraction']:.1%}{tags}")
            if region["percentage_error"] is not None and region["percentage_error"] > 10:
                flags.append(f"{label}/{region['name']}: depth error={region['percentage_error']:.1f}%{tags}")
    for scene in report["scene_metrics"]:
        for pair in scene["occupancy_stability"]:
            if pair["occupied_iou"] < 0.5:
                flags.append(f"{scene['scene_id']} {pair['previous_frame_id']}→{pair['frame_id']}: occupancy IoU={pair['occupied_iou']:.2f}")
    return flags


def _resolve_dataset_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"dataset path escapes manifest directory: {value}") from exc
    return path


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _percentile_or_none(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _format_number(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def _format_percent_fraction(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
