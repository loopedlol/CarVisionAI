"""Comparative OpenCV diagnostics for timestamped pose estimators."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .control import wrap_angle
from .fusion_simulation import FusionSimulationResult
from .pose_estimation import EstimatorMode


COLORS = [(40, 80, 210), (20, 150, 60), (180, 60, 160)]
MODE_COLOR = dict(zip(EstimatorMode, COLORS))


def render_fusion_comparison(results: tuple[FusionSimulationResult, ...], title: str,
                             *, width: int = 1600, height: int = 1000) -> NDArray[np.uint8]:
    image = np.full((height, width, 3), 248, np.uint8)
    cv2.putText(image, title, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    _paths(image, (25, 60, 700, 900), results)
    series = []
    for result in results:
        time = np.asarray([f.timestamp_s for f in result.feedback])
        position_error = np.asarray([np.hypot(s.pose.x_m - t.x_m, s.pose.y_m - t.y_m)
                                     for s, t in zip(result.estimator_status, result.truths)])
        heading_error = np.asarray([abs(wrap_angle(s.pose.heading_rad - t.heading_rad))
                                    for s, t in zip(result.estimator_status, result.truths)])
        sigma_position = np.asarray([np.sqrt(max(s.covariance[0, 0], s.covariance[1, 1]))
                                     for s in result.estimator_status])
        series.append((time, position_error, heading_error, sigma_position))
    colors = [MODE_COLOR[result.mode] for result in results]
    _multi_chart(image, (760, 60, 810, 185), series, colors, 1, "position error m")
    _multi_chart(image, (760, 265, 810, 185), series, colors, 2, "absolute heading error rad")
    _multi_chart(image, (760, 470, 810, 185), series, colors, 3, "reported 1-sigma position uncertainty m")
    primary = results[-1]
    time = np.asarray([f.timestamp_s for f in primary.feedback])
    imu = np.asarray(primary.imu_measurements_rps)
    _single_chart(image, (760, 675, 810, 150), time, imu, "IMU yaw-rate measurement rad/s")
    for index, accepted in primary.correction_events:
        if index >= len(time): continue
        px = 760 + int(index / max(len(time) - 1, 1) * 810)
        cv2.line(image, (px, 675), (px, 825), (20, 150, 20) if accepted else (20, 20, 220), 1)
    _dropout_bar(image, (760, 840, 810, 35), primary.encoder_dropout_mask,
                 primary.imu_dropout_mask)
    y = 900
    for result, color in zip(results, colors):
        m = result.metrics
        text = (f"{result.mode.value}: RMSE={m.position_rmse_m:.3f}m final={m.final_position_error_m:.3f}m "
                f"heading_RMSE={m.heading_rmse_rad:.3f}rad corrections={m.accepted_corrections}/"
                f"{m.rejected_corrections} health={m.final_health.value} controller={m.controller_state.value}")
        cv2.putText(image, text, (760, y), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1); y += 25
    return image


def _paths(image: NDArray[np.uint8], box: tuple[int, int, int, int],
           results: tuple[FusionSimulationResult, ...]) -> None:
    x, y, w, h = box; cv2.rectangle(image, (x, y), (x + w, y + h), (210, 210, 210), 1)
    truths = [np.asarray([(p.x_m, p.y_m) for p in result.truths]) for result in results]
    paths = [np.asarray([(s.pose.x_m, s.pose.y_m) for s in result.estimator_status])
             for result in results]
    values = np.vstack([*truths, *paths]); low, high = values.min(0), values.max(0)
    span = np.maximum(high - low, .2); low -= span * .08; high += span * .08
    def px(points: NDArray) -> NDArray[np.int32]:
        return np.column_stack((x + 15 + (points[:, 0] - low[0]) / (high[0] - low[0]) * (w - 30),
                                y + h - 15 - (points[:, 1] - low[1]) / (high[1] - low[1]) * (h - 30))).astype(np.int32)
    colors = [MODE_COLOR[result.mode] for result in results]
    for truth, color in zip(truths, colors):
        dark = tuple(int(channel * .45) for channel in color)
        cv2.polylines(image, [px(truth)], False, dark, 4)
    for result, path, color in zip(results, paths, colors):
        cv2.polylines(image, [px(path)], False, color, 2)
    cv2.putText(image, "each run: truth=dark/thick, estimate=bright/thin; modes red / green / purple",
                (x + 12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1)


def _multi_chart(image: NDArray[np.uint8], box: tuple[int, int, int, int], series: list,
                 colors: list[tuple[int, int, int]], field: int, label: str) -> None:
    arrays = [item[field] for item in series]; maximum = max(float(np.max(a)) for a in arrays)
    _chart_frame(image, box, label)
    for item, color in zip(series, colors): _plot(image, box, item[0], item[field], 0, max(maximum, 1e-6), color)


def _single_chart(image: NDArray[np.uint8], box: tuple[int, int, int, int], time: NDArray,
                  values: NDArray, label: str) -> None:
    _chart_frame(image, box, label)
    finite = values[np.isfinite(values)]
    bound = max(float(np.max(np.abs(finite))) if len(finite) else 1, .1)
    _plot(image, box, time, np.nan_to_num(values), -bound, bound, (180, 80, 20))


def _chart_frame(image: NDArray[np.uint8], box: tuple[int, int, int, int], label: str) -> None:
    x, y, w, h = box; cv2.rectangle(image, (x, y), (x + w, y + h), (210, 210, 210), 1)
    cv2.putText(image, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (40, 40, 40), 1)


def _plot(image: NDArray[np.uint8], box: tuple[int, int, int, int], time: NDArray,
          values: NDArray, low: float, high: float, color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    px = x + (time - time[0]) / max(time[-1] - time[0], 1e-9) * w
    py = y + h - (values - low) / (high - low) * (h - 25)
    cv2.polylines(image, [np.column_stack((px, py)).astype(np.int32)], False, color, 2)


def _dropout_bar(image: NDArray[np.uint8], box: tuple[int, int, int, int], enc: tuple[bool, ...],
                 imu: tuple[bool, ...]) -> None:
    x, y, w, h = box; cv2.putText(image, "dropout: encoder top / IMU bottom", (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, .38, (40, 40, 40), 1)
    for index, missing in enumerate(enc):
        if missing: cv2.line(image, (x + index * w // len(enc), y),
                            (x + index * w // len(enc), y + h // 2), (20, 20, 220), 1)
    for index, missing in enumerate(imu):
        if missing: cv2.line(image, (x + index * w // len(imu), y + h // 2),
                            (x + index * w // len(imu), y + h), (180, 80, 20), 1)
