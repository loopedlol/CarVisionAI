"""OpenCV diagnostics for closed-loop simulation results."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from .control import SimulationResult
from .trajectory import GeneratedTrajectory


def render_simulation(trajectory: GeneratedTrajectory, result: SimulationResult,
                      title: str, *, width: int = 1500, height: int = 950) -> NDArray[np.uint8]:
    canvas = np.full((height, width, 3), 248, np.uint8)
    cv2.putText(canvas, title, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (30, 30, 30), 2)
    feedback = result.feedback
    truth = np.asarray([(f.ground_truth_pose.x_m, f.ground_truth_pose.y_m) for f in feedback])
    estimate = np.asarray([(f.estimated_pose.x_m, f.estimated_pose.y_m) for f in feedback])
    planned = np.asarray([(s.x_m, s.y_m) for s in trajectory.samples])
    _path_panel(canvas, (30, 60, 700, 880), planned, truth, estimate)
    time = np.asarray([f.timestamp_s for f in feedback])
    target_v = np.asarray([f.commanded_linear_mps for f in feedback])
    target_w = np.asarray([f.commanded_angular_rps for f in feedback])
    actual_v = np.asarray(result.actual_linear_mps[:len(feedback)])
    actual_w = np.asarray(result.actual_angular_rps[:len(feedback)])
    left = np.asarray([f.commanded_wheels.left_rad_s for f in feedback])
    right = np.asarray([f.commanded_wheels.right_rad_s for f in feedback])
    headings = np.asarray([f.ground_truth_pose.heading_rad for f in feedback])
    position = np.asarray([f.position_error_m for f in feedback])
    heading_error = np.asarray([f.heading_error_rad for f in feedback])
    cross = np.asarray([f.cross_track_error_m for f in feedback])
    panels = [
        ("linear velocity m/s", target_v, actual_v, (30, 80, 210), (20, 150, 40)),
        ("angular velocity rad/s", target_w, actual_w, (30, 80, 210), (20, 150, 40)),
        ("wheel commands rad/s", left, right, (180, 60, 30), (20, 130, 190)),
        ("heading / heading error rad", headings, heading_error, (100, 30, 180), (30, 100, 220)),
        ("position / cross-track error m", position, cross, (180, 80, 30), (30, 150, 80)),
    ]
    for i, panel in enumerate(panels):
        y = 60 + i * 160
        _chart(canvas, (750, y, 720, 140), time, panel[1], panel[2], panel[0], panel[3], panel[4])
    last = feedback[-1]
    summary = (f"state={last.state.value} fault={last.fault.value} steps={len(feedback)} "
               f"runtime={result.runtime_ms:.1f}ms controller_p50="
               f"{np.median(result.controller_update_ms):.3f}ms")
    cv2.putText(canvas, summary, (750, 890), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1)
    return canvas


def _path_panel(image: NDArray[np.uint8], box: tuple[int, int, int, int], planned: NDArray,
                truth: NDArray, estimate: NDArray) -> None:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (210, 210, 210), 1)
    all_points = np.vstack((planned, truth, estimate))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    span = np.maximum(high - low, .2); low -= span * .1; high += span * .1
    def pixels(values: NDArray) -> NDArray[np.int32]:
        px = x + 20 + (values[:, 0] - low[0]) / (high[0] - low[0]) * (w - 40)
        py = y + h - 20 - (values[:, 1] - low[1]) / (high[1] - low[1]) * (h - 40)
        return np.column_stack((px, py)).astype(np.int32)
    cv2.polylines(image, [pixels(planned)], False, (80, 80, 80), 3)
    cv2.polylines(image, [pixels(truth)], False, (40, 150, 40), 2)
    cv2.polylines(image, [pixels(estimate)], False, (200, 90, 30), 1)
    cv2.putText(image, "planned=gray truth=green estimate=blue", (x + 15, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (40, 40, 40), 1)


def _chart(image: NDArray[np.uint8], box: tuple[int, int, int, int], time: NDArray,
           first: NDArray, second: NDArray, label: str, first_color: tuple[int, int, int],
           second_color: tuple[int, int, int]) -> None:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (210, 210, 210), 1)
    cv2.putText(image, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (40, 40, 40), 1)
    if not len(time): return
    low = min(float(first.min()), float(second.min()), 0.0)
    high = max(float(first.max()), float(second.max()), 0.0)
    if high - low < 1e-6: high = low + 1
    def line(values: NDArray) -> NDArray[np.int32]:
        px = x + 8 + (time - time[0]) / max(time[-1] - time[0], 1e-6) * (w - 16)
        py = y + h - 8 - (values - low) / (high - low) * (h - 32)
        return np.column_stack((px, py)).astype(np.int32)
    cv2.polylines(image, [line(first)], False, first_color, 2)
    cv2.polylines(image, [line(second)], False, second_color, 1)
