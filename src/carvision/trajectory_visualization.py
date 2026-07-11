"""Diagnostics for geometric paths, speed profiles, swept footprints, and updates."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .mapping import FREE, OCCUPIED, UNKNOWN
from .planning import PreparedPlanningGrid
from .trajectory import GeneratedTrajectory, TrajectoryConfig, TrajectoryValidation


def render_trajectory(prepared: PreparedPlanningGrid, trajectory: GeneratedTrajectory,
                      trajectory_config: TrajectoryConfig, *,
                      validation: TrajectoryValidation | None = None,
                      changed_mask: NDArray[np.bool_] | None = None,
                      current_index: int = 0, scale: int = 7) -> NDArray[np.uint8]:
    data = prepared.data
    image = np.zeros((*data.shape, 3), np.uint8)
    image[data == FREE] = (240, 240, 240)
    image[data == UNKNOWN] = (130, 130, 130)
    image[data == OCCUPIED] = (20, 20, 20)
    image[prepared.inflated_unsafe & (data != OCCUPIED)] = (30, 130, 240)
    if changed_mask is not None:
        image[np.asarray(changed_mask, bool)] = (40, 210, 40)
    image = cv2.resize(np.flipud(image), (data.shape[1] * scale, data.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    _polyline(image, trajectory.raw_metric_path, prepared, scale, (180, 80, 200), 1)
    _polyline(image, trajectory.simplified_path, prepared, scale, (255, 120, 20), 2)
    _polyline(image, trajectory.smoothed_path, prepared, scale, (255, 255, 0), 2)
    speeds = np.asarray([sample.linear_velocity_mps for sample in trajectory.samples])
    maximum = max(float(np.max(speeds)), 1e-6)
    for index, sample in enumerate(trajectory.samples):
        point = _metric_pixel(sample.x_m, sample.y_m, prepared, scale)
        color_value = int(np.clip(255 * sample.linear_velocity_mps / maximum, 0, 255))
        color = (255 - color_value, 40, color_value)
        cv2.circle(image, point, 2, color, -1)
        if index % max(1, len(trajectory.samples) // 24) == 0:
            length = scale * 1.6
            end = (round(point[0] + np.sin(sample.heading_rad) * length),
                   round(point[1] - np.cos(sample.heading_rad) * length))
            cv2.arrowedLine(image, point, end, (30, 30, 30), 1, tipLength=0.3)
    radius_px = max(2, round(trajectory_config.required_clearance_m
                             / prepared.config.resolution_m * scale))
    for sample in trajectory.samples[::max(1, len(trajectory.samples) // 18)]:
        cv2.circle(image, _metric_pixel(sample.x_m, sample.y_m, prepared, scale),
                   radius_px, (170, 170, 40), 1, cv2.LINE_AA)
    for corner in trajectory.rejected_smoothing_corners:
        if corner < len(trajectory.simplified_path):
            point = _metric_pixel(*trajectory.simplified_path[corner], prepared, scale)
            cv2.drawMarker(image, point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    if validation is not None:
        current_distance = trajectory.samples[current_index].distance_m
        for sample in trajectory.samples[current_index:]:
            if sample.distance_m - current_distance > validation.braking_distance_m:
                break
            cv2.circle(image, _metric_pixel(sample.x_m, sample.y_m, prepared, scale),
                       4, (0, 0, 255), -1)
    panel = np.full((78, image.shape[1], 3), 250, np.uint8)
    cv2.putText(panel, "raw=purple simplified=orange smooth=cyan speed=blue->red swept=olive",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(panel, f"samples={len(trajectory.samples)} fallback={trajectory.used_fallback} "
                f"rejected_corners={trajectory.rejected_smoothing_corners}",
                (8, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
    if validation is not None:
        cv2.putText(panel, f"action={validation.action.value} braking={validation.braking_distance_m:.2f}m "
                    f"reason={validation.reasons[0][:60]}", (8, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack((panel, image))


def _polyline(image: NDArray[np.uint8], points: NDArray[np.float64],
              prepared: PreparedPlanningGrid, scale: int,
              color: tuple[int, int, int], thickness: int) -> None:
    pixels = np.asarray([_metric_pixel(point[0], point[1], prepared, scale)
                         for point in points], np.int32)
    cv2.polylines(image, [pixels], False, color, thickness, cv2.LINE_AA)


def _metric_pixel(x_m: float, y_m: float, prepared: PreparedPlanningGrid,
                  scale: int) -> tuple[int, int]:
    config = prepared.config
    column = (x_m - config.x_min_m) / config.resolution_m
    row = (y_m - config.y_min_m) / config.resolution_m
    return round(column * scale), round((config.height - row) * scale)
