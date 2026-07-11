"""Deterministic closed-loop disturbance presets for tests and demonstrations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from .control import VehicleConfig


SCENARIOS = ("ideal", "command_delay", "asymmetric_motors", "wheel_saturation",
             "encoder_noise", "moderate_slip", "severe")


def path_scenario(name: str, count: int = 61) -> NDArray[np.float64]:
    """Reference geometry for controller-only straight/turn/S tests."""
    t = np.linspace(0, 1, count)
    if name == "straight":
        return np.column_stack((np.zeros_like(t), 3 * t))
    if name == "gradual_turn":
        angle = np.pi * t / 2
        return np.column_stack((2 * (1 - np.cos(angle)), 2 * np.sin(angle)))
    if name == "sharp_turn":
        first = np.column_stack((np.zeros(count // 2), np.linspace(0, 1.5, count // 2)))
        second = np.column_stack((np.linspace(0, 1.5, count - len(first)),
                                  np.full(count - len(first), 1.5)))
        return np.vstack((first, second[1:]))
    if name == "s_curve":
        return np.column_stack((.45 * np.sin(2 * np.pi * t), 3 * t))
    raise ValueError("path scenario must be straight, gradual_turn, sharp_turn, or s_curve")


def vehicle_scenario(name: str, base: VehicleConfig = VehicleConfig()) -> VehicleConfig:
    if name == "ideal":
        return base
    if name == "command_delay":
        return replace(base, command_delay_s=0.15)
    if name == "asymmetric_motors":
        return replace(base, left_motor_scale=0.94, right_motor_scale=1.02)
    if name == "wheel_saturation":
        return replace(base, max_wheel_speed_rad_s=5.0)
    if name == "encoder_noise":
        return replace(base, encoder_noise_std_rad_s=0.10,
                       pose_position_noise_std_m=0.003,
                       pose_heading_noise_std_rad=0.006)
    if name == "moderate_slip":
        return replace(base, command_delay_s=0.08, left_motor_scale=0.97,
                       right_motor_scale=1.01, encoder_noise_std_rad_s=0.06,
                       pose_position_noise_std_m=0.002,
                       pose_heading_noise_std_rad=0.004,
                       longitudinal_slip=0.06, angular_slip=0.04,
                       motor_response_scale=0.96)
    if name == "severe":
        return replace(base, command_delay_s=0.18, left_motor_scale=0.45,
                       right_motor_scale=1.0, encoder_noise_std_rad_s=0.12,
                       longitudinal_slip=0.25, angular_slip=0.15,
                       max_wheel_speed_rad_s=5.0, motor_response_scale=0.85)
    raise ValueError(f"unknown control scenario {name!r}; choose from {SCENARIOS}")
