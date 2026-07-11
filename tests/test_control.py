from dataclasses import replace
from math import pi

import numpy as np
import pytest

from carvision.control import (
    ControllerConfig, ControllerState, DifferentialDriveSimulator, FaultCode, Pose2D,
    TrajectoryCommand, TrajectoryFollower, VehicleConfig, WheelSpeeds, body_to_wheels,
    integrate_unicycle, simulate_trajectory, wheels_to_body, wrap_angle,
)
from carvision.trajectory import GeneratedTrajectory, TrajectorySample


def trajectory(points: list[tuple[float, float]], final_heading: float | None = None,
               speed: float = 0.45) -> GeneratedTrajectory:
    values = np.asarray(points, dtype=float)
    distances = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(values, axis=0), axis=1))))
    samples = []
    for i, (x, y) in enumerate(values):
        if i < len(values) - 1:
            dx, dy = values[i + 1] - values[i]
            heading = np.arctan2(dx, dy)
        else:
            heading = final_heading if final_heading is not None else samples[-1].heading_rad
        samples.append(TrajectorySample(x, y, heading, 0, 2, False,
                                        0 if i == len(values) - 1 else speed, 0,
                                        distances[i] / speed, distances[i]))
    return GeneratedTrajectory(values, values, values, tuple(samples), False, False, (), {})


def command(points=((0, 0), (0, 0.5), (0, 1.0)), **kwargs) -> TrajectoryCommand:
    return TrajectoryCommand(7, 3, trajectory(list(points), **kwargs))


def test_body_wheel_conversion_round_trip_and_sign_convention():
    wheels = body_to_wheels(0.6, 0.7, 0.1, 0.4)
    assert wheels.left_rad_s > wheels.right_rad_s
    assert wheels_to_body(wheels, 0.1, 0.4) == pytest.approx((0.6, 0.7))


def test_ideal_straight_and_arc_kinematic_integration():
    assert integrate_unicycle(Pose2D(0, 0, 0), 1, 0, 2) == Pose2D(0, 2, 0)
    arc = integrate_unicycle(Pose2D(0, 0, 0), 1, 1, pi / 2)
    assert (arc.x_m, arc.y_m, arc.heading_rad) == pytest.approx((1, 1, pi / 2))


def test_heading_wrap():
    assert wrap_angle(3 * pi) == pytest.approx(-pi)
    assert wrap_angle(-3 * pi / 2) == pytest.approx(pi / 2)


def test_progress_never_moves_backwards():
    follower = TrajectoryFollower(VehicleConfig())
    cmd = command(points=((0, 0), (0, .2), (0, .4), (0, .6)))
    follower.accept(cmd, 3)
    first = follower.update(Pose2D(0, .42, 0), 0, current_map_revision=3)
    second = follower.update(Pose2D(0, .18, 0), .04, current_map_revision=3)
    assert first.progress_index >= 2
    assert second.progress_index >= first.progress_index


def test_controller_steers_toward_target_on_right():
    follower = TrajectoryFollower(VehicleConfig())
    cmd = command(points=((0, 0), (.2, .4), (.4, .8)))
    follower.accept(cmd, 3)
    feedback = follower.update(Pose2D(0, 0, 0), 0, current_map_revision=3)
    assert feedback.commanded_angular_rps > 0
    assert feedback.commanded_wheels.left_rad_s > feedback.commanded_wheels.right_rad_s


def test_speed_and_acceleration_limits():
    vehicle = VehicleConfig(control_frequency_hz=10, max_linear_acceleration_mps2=.5,
                            max_angular_acceleration_rps2=1)
    follower = TrajectoryFollower(vehicle)
    cmd = command(points=((0, 0), (.5, .5), (1, 1)), speed=5)
    follower.accept(cmd, 3)
    feedback = follower.update(Pose2D(0, 0, 0), 0, current_map_revision=3)
    assert feedback.commanded_linear_mps <= .05 + 1e-9
    assert abs(feedback.commanded_angular_rps) <= .1 + 1e-9


def test_stale_and_revision_mismatch_rejected():
    follower = TrajectoryFollower(VehicleConfig())
    assert not follower.accept(command(), 3, newest_trajectory_id=8)
    assert follower.fault == FaultCode.STALE_TRAJECTORY
    assert not follower.accept(command(), 4)
    assert follower.fault == FaultCode.MAP_REVISION_MISMATCH


def test_cancellation_invalidation_emergency_and_watchdog():
    follower = TrajectoryFollower(VehicleConfig(), ControllerConfig(watchdog_timeout_s=.1))
    follower.accept(command(), 3)
    follower.update(Pose2D(0, 0, 0), 0, current_map_revision=3)
    follower.cancel(invalidated=True)
    assert follower.fault == FaultCode.INVALIDATED
    assert follower.update(Pose2D(0, 0, 0), .01, current_map_revision=3).commanded_linear_mps == 0
    follower.accept(command(), 3)
    follower.update(Pose2D(0, 0, 0), 0, current_map_revision=3)
    assert follower.watchdog(.2, Pose2D(0, 0, 0)).state == ControllerState.WATCHDOG_STOP
    follower.accept(command(), 3)
    follower.emergency_stop()
    assert follower.fault == FaultCode.EMERGENCY_STOP


def test_excessive_tracking_error_fault():
    follower = TrajectoryFollower(VehicleConfig(), ControllerConfig(max_cross_track_error_m=.2))
    follower.accept(command(), 3)
    result = follower.update(Pose2D(1, 0, 0), 0, current_map_revision=3)
    assert result.fault == FaultCode.CROSS_TRACK_ERROR
    assert result.commanded_linear_mps == 0


def test_saturation_persistence_fault():
    vehicle = VehicleConfig(max_wheel_speed_rad_s=1)
    follower = TrajectoryFollower(vehicle, ControllerConfig(saturation_fault_steps=2))
    follower.accept(command(), 3)
    follower.update(Pose2D(0, 0, 0), 0, current_map_revision=3, wheel_saturated=True)
    result = follower.update(Pose2D(0, .01, 0), vehicle.dt_s,
                             current_map_revision=3, wheel_saturated=True)
    assert result.fault == FaultCode.PERSISTENT_SATURATION


def test_terminal_heading_alignment_and_stop():
    follower = TrajectoryFollower(VehicleConfig())
    cmd = command(points=((0, 0), (0, .5)), final_heading=pi / 2)
    follower.accept(cmd, 3)
    turning = follower.update(Pose2D(0, .5, 0), 0, current_map_revision=3)
    assert turning.state == ControllerState.TERMINAL_ALIGN
    assert turning.commanded_angular_rps > 0
    stopped = follower.update(Pose2D(0, .5, pi / 2), .1, current_map_revision=3)
    assert stopped.state == ControllerState.STOPPED
    assert stopped.commanded_angular_rps == 0


def test_ideal_closed_loop_reaches_terminal_stop():
    result = simulate_trajectory(command(points=tuple((0, y) for y in np.linspace(0, 2, 41))),
                                 VehicleConfig(), max_time_s=15)
    assert result.feedback[-1].state == ControllerState.STOPPED
    assert result.feedback[-1].ground_truth_pose.y_m == pytest.approx(2, abs=.08)


def test_simulation_is_deterministic_and_disturbance_creates_odometry_error():
    cmd = command(points=tuple((0, y) for y in np.linspace(0, 2, 41)))
    config = VehicleConfig(encoder_noise_std_rad_s=.08, left_motor_scale=.94,
                           longitudinal_slip=.08, pose_position_noise_std_m=.002)
    first = simulate_trajectory(cmd, config, seed=12, max_time_s=15)
    second = simulate_trajectory(cmd, config, seed=12, max_time_s=15)
    a, b = first.feedback[-1], second.feedback[-1]
    assert a.ground_truth_pose == b.ground_truth_pose
    error = np.hypot(a.ground_truth_pose.x_m - a.estimated_pose.x_m,
                     a.ground_truth_pose.y_m - a.estimated_pose.y_m)
    assert error > .02


def test_severe_disturbance_and_saturation_trigger_safe_failure():
    cmd = command(points=tuple((0, y) for y in np.linspace(0, 2, 41)))
    config = VehicleConfig(left_motor_scale=.4, right_motor_scale=1,
                           longitudinal_slip=.25, angular_slip=.1,
                           max_wheel_speed_rad_s=5)
    result = simulate_trajectory(cmd, config, ControllerConfig(max_cross_track_error_m=.25,
                                                                saturation_fault_steps=20),
                                 seed=1, max_time_s=15)
    assert result.feedback[-1].state == ControllerState.FAULT
    assert result.feedback[-1].fault == FaultCode.PERSISTENT_SATURATION


def test_encoder_noise_has_known_zero_mean_scale():
    config = VehicleConfig(control_frequency_hz=50, encoder_noise_std_rad_s=.2)
    sim = DifferentialDriveSimulator(config, Pose2D(0, 0, 0), seed=42)
    measured = [sim.step(WheelSpeeds(4, 4))[2].left_rad_s for _ in range(2000)]
    assert np.mean(measured) == pytest.approx(4, abs=.02)
    assert np.std(measured) == pytest.approx(.2, rel=.12)
