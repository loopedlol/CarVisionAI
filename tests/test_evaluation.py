import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from carvision.evaluation import (
    EvaluationDataset,
    EvaluationRegion,
    compare_configurations,
    occupancy_metrics,
    occupancy_pair_stability,
    region_depth_metrics,
    render_markdown_report,
    sgbm_config_from_dict,
    summarize_evaluation,
    temporal_region_metrics,
    validity_metrics,
    vertical_rectification_metrics,
)
from carvision.frames import CAMERA_OPTICAL_TO_VEHICLE
from carvision.mapping import FREE, OCCUPIED, UNKNOWN
from carvision.recorded_stereo import SGBMConfig, StereoRigCalibration


def write_calibration(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    width, height = size
    matrix = np.array([[50.0, 0, (width - 1) / 2], [0, 50.0, (height - 1) / 2], [0, 0, 1]])
    StereoRigCalibration(size, matrix, np.zeros(5), matrix, np.zeros(5),
                         np.eye(3), np.array([-0.1, 0, 0]),
                         CAMERA_OPTICAL_TO_VEHICLE).save(path)


def write_dataset(tmp_path: Path, *, bad_roi: bool = False, missing: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    write_calibration(tmp_path / "calibration.json")
    (tmp_path / "left").mkdir(); (tmp_path / "right").mkdir()
    image = np.zeros((48, 64), np.uint8)
    cv2.imwrite(str(tmp_path / "left" / "000.png"), image)
    if not missing:
        cv2.imwrite(str(tmp_path / "right" / "000.png"), image)
    manifest = {
        "format_version": 1, "name": "fixture", "calibration": "calibration.json",
        "expected_image_size": [64, 48], "default_sgbm": {"num_disparities": 16},
        "scenes": [{"id": "static_wall", "category": "indoor", "static": True,
                    "conditions": {"lighting": "even", "texture": "low"},
                    "frames": [{"id": "000", "left": "left/000.png", "right": "right/000.png",
                                "regions": [{"name": "wall", "rect": [50, 10, 20, 10] if bad_roi else [20, 10, 20, 10],
                                             "expected_depth_m": 2.0,
                                             "tags": ["textureless"]}]}]}],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest))
    return path


def test_dataset_parser_resolves_paths_and_annotations(tmp_path: Path) -> None:
    dataset = EvaluationDataset.load(write_dataset(tmp_path))
    assert dataset.name == "fixture"
    scene, frame = dataset.find("static_wall", "000")
    assert scene.static and scene.conditions["texture"] == "low"
    assert frame.left_path.is_absolute()
    assert frame.regions[0].expected_depth_m == 2.0
    assert frame.regions[0].tags == ("textureless",)


def test_dataset_parser_rejects_missing_images_bad_roi_and_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing image"):
        EvaluationDataset.load(write_dataset(tmp_path / "missing", missing=True))
    with pytest.raises(ValueError, match="exceeds"):
        EvaluationDataset.load(write_dataset(tmp_path / "roi", bad_roi=True))
    root = tmp_path / "escape"; root.mkdir()
    write_calibration(root / "calibration.json")
    cv2.imwrite(str(tmp_path / "outside.png"), np.zeros((48, 64), np.uint8))
    manifest = {"format_version": 1, "name": "bad", "calibration": "calibration.json",
                "expected_image_size": [64, 48], "scenes": [
                    {"id": "s", "frames": [{"id": "f", "left": "../outside.png",
                                               "right": "../outside.png"}]}]}
    (root / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        EvaluationDataset.load(root / "dataset.json")


def test_validity_metrics_calculates_search_border_separately() -> None:
    mask = np.ones((2, 10), bool)
    mask[:, :3] = False
    mask[0, 8] = False
    metrics = validity_metrics(mask, 3)
    assert metrics["valid_disparity_fraction"] == pytest.approx(13 / 20)
    assert metrics["invalid_border_fraction"] == 1.0
    assert metrics["invalid_pixels_in_search_border_fraction"] == pytest.approx(6 / 7)


def test_region_depth_robust_statistics_and_measured_error() -> None:
    depth = np.full((10, 10), np.nan)
    depth[2:6, 3:7] = 2.2
    depth[2, 3] = 20.0
    region = EvaluationRegion("target", (3, 2, 4, 4), 2.0, ("reflective",))
    metrics = region_depth_metrics(depth, region)
    assert metrics["valid_fraction"] == 1.0
    assert metrics["depth_median_m"] == pytest.approx(2.2)
    assert metrics["absolute_error_m"] == pytest.approx(0.2)
    assert metrics["percentage_error"] == pytest.approx(10.0)
    assert metrics["depth_mad_m"] == pytest.approx(0.0)


def test_rectification_metric_detects_known_vertical_offset() -> None:
    rng = np.random.default_rng(9)
    left = cv2.GaussianBlur(rng.integers(0, 256, (160, 240), np.uint8), (3, 3), 0)
    right = np.zeros_like(left)
    # Right content is eight pixels left and two pixels down.
    right[2:, :-8] = left[:-2, 8:]
    metrics = vertical_rectification_metrics(left, right)
    assert metrics["feature_match_count"] > 100
    assert metrics["vertical_error_median_px"] == pytest.approx(2.0, abs=0.4)


def test_occupancy_components_and_pair_stability() -> None:
    first = np.full((5, 6), UNKNOWN, np.int8)
    first[1, 1:4] = OCCUPIED
    first[4, 5] = OCCUPIED
    first[0, 0] = FREE
    metrics = occupancy_metrics(first, isolated_max_cells=1)
    assert metrics["occupied_component_count"] == 2
    assert metrics["isolated_component_count"] == 1
    second = first.copy(); second[1, 1] = FREE; second[2, 3] = OCCUPIED
    stability = occupancy_pair_stability(first, second)
    assert stability["occupied_intersection_cells"] == 3
    assert stability["occupied_union_cells"] == 5
    assert stability["occupied_iou"] == pytest.approx(0.6)


def test_temporal_metrics_use_per_frame_region_medians() -> None:
    frames = [{"regions": [{"name": "wall", "depth_median_m": value}]}
              for value in (2.0, 2.1, 1.9)]
    temporal = temporal_region_metrics(frames)[0]
    assert temporal["median_depth_m"] == 2.0
    assert temporal["temporal_range_m"] == pytest.approx(0.2)
    assert temporal["sample_count"] == 3


def test_config_override_validation_and_comparison_bound(tmp_path: Path) -> None:
    config = sgbm_config_from_dict({"block_size": 7, "num_disparities": 32})
    assert config.block_size == 7
    with pytest.raises(ValueError, match="unknown"):
        sgbm_config_from_dict({"magic": 1})
    dataset = EvaluationDataset.load(write_dataset(tmp_path))
    with pytest.raises(ValueError, match="between 1 and 12"):
        compare_configurations(dataset, {}, tmp_path / "out")


def test_summary_and_markdown_flag_tradeoffs() -> None:
    frames = [{"scene_id": "s", "frame_id": "f", "valid_disparity_fraction": 0.1,
               "vertical_error_p95_px": 2.0, "timings_ms": {"total": 12.0},
               "regions": [{"name": "target", "percentage_error": 20.0}]}]
    scenes = [{"scene_id": "s", "occupancy_stability": [
        {"previous_frame_id": "a", "frame_id": "f", "occupied_iou": 0.2}]}]
    summary = summarize_evaluation(frames, scenes)
    report = {"dataset": "fixture", "summary": summary,
              "frame_metrics": frames, "scene_metrics": scenes}
    markdown = render_markdown_report(report)
    assert "poor rectification" in markdown
    assert "depth error" in markdown
    assert "occupancy IoU" in markdown
