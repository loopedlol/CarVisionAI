from math import pi

import numpy as np
import pytest

from carvision.control import Pose2D, WheelSpeeds
from carvision.control import TrajectoryCommand, VehicleConfig
from carvision.fusion_simulation import SensorSimulationConfig, simulate_fused_trajectory
from carvision.pose_estimation import (
    EncoderMeasurement, EstimatorConfig, EstimatorFault, EstimatorMode,
    ExternalPoseMeasurement, HeadingMeasurement, ImuYawRateMeasurement, PoseEstimator,
)
from carvision.trajectory import GeneratedTrajectory, TrajectorySample


def encoder(time: float, left: float, right: float | None = None) -> EncoderMeasurement:
    return EncoderMeasurement(time, WheelSpeeds(left, left if right is None else right))


def test_timestamp_order_and_bounded_delayed_imu():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU)
    assert estimator.process_encoder(encoder(0, 4))
    assert estimator.process_encoder(encoder(.1, 4))
    assert estimator.process_imu(ImuYawRateMeasurement(.05, 0))
    assert not estimator.process_imu(ImuYawRateMeasurement(.04, 0))
    assert estimator.status().fault == EstimatorFault.TIMESTAMP_DISORDER
    assert not estimator.process_encoder(encoder(.05, 4))


def test_differential_drive_prediction_straight_and_turn():
    config = EstimatorConfig(wheel_radius_m=.1, track_width_m=.5)
    straight = PoseEstimator(EstimatorMode.ENCODER_ONLY, config)
    straight.process_encoder(encoder(0, 10)); straight.process_encoder(encoder(1, 10))
    assert straight.pose == Pose2D(0, 1, 0)
    turning = PoseEstimator(EstimatorMode.ENCODER_ONLY, config)
    turning.process_encoder(encoder(0, 12.5, 7.5))
    turning.process_encoder(encoder(1, 12.5, 7.5))
    assert turning.pose.heading_rad == pytest.approx(1)
    assert turning.pose.x_m > 0


def test_angle_wrapping_in_prediction_and_heading_update():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU)
    estimator.reset(Pose2D(0, 0, pi - .01), 0)
    estimator.process_encoder(encoder(0, 1))
    estimator.process_imu(ImuYawRateMeasurement(0, .5))
    estimator.process_encoder(encoder(.1, 1))
    assert -pi <= estimator.pose.heading_rad < pi
    assert estimator.process_heading(HeadingMeasurement(.1, -pi + .02, .01))


def test_covariance_propagates_and_grows_during_dropout():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU)
    before = np.trace(estimator.covariance)
    estimator.process_encoder(encoder(0, 4)); estimator.process_encoder(encoder(.1, 4))
    propagated = np.trace(estimator.covariance)
    estimator.advance_time(2.0)
    assert propagated > before
    assert np.trace(estimator.covariance) > propagated


def test_imu_yaw_rate_changes_future_heading_prediction():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU)
    estimator.process_encoder(encoder(0, 5))
    estimator.process_imu(ImuYawRateMeasurement(.03, .2, .0001))
    estimator.process_encoder(encoder(.1, 5))
    assert estimator.pose.heading_rad > .01


def test_external_pose_update_and_outlier_gate():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU_EXTERNAL)
    estimator.covariance[:3, :3] = np.eye(3) * .2
    accepted = estimator.process_external_pose(
        ExternalPoseMeasurement(0, Pose2D(.2, .1, .05), np.eye(3) * .01))
    assert accepted and estimator.pose.x_m > .1
    old = estimator.pose
    assert not estimator.process_external_pose(
        ExternalPoseMeasurement(.1, Pose2D(100, 100, 2), np.eye(3) * .01))
    assert estimator.pose == old
    assert estimator.status().rejected_outliers == 1


def test_reset_clears_counts_state_and_covariance():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU)
    estimator.process_encoder(encoder(0, 4)); estimator.process_encoder(encoder(.1, 4))
    estimator.reset(Pose2D(2, 3, 4), 5)
    status = estimator.status()
    assert status.pose == Pose2D(2, 3, pytest.approx(4 - 2 * pi))
    assert status.accepted_encoder == 0
    assert status.timestamp_s == 5


def test_nonfinite_and_invalid_covariance_rejected():
    estimator = PoseEstimator(EstimatorMode.ENCODER_IMU_EXTERNAL)
    assert not estimator.process_imu(ImuYawRateMeasurement(0, np.nan))
    assert estimator.status().fault == EstimatorFault.NONFINITE_MEASUREMENT
    with pytest.raises(ValueError):
        estimator.process_external_pose(ExternalPoseMeasurement(0, Pose2D(0, 0, 0),
                                                                 np.zeros((3, 3))))


def test_deterministic_measurement_sequence():
    def run():
        estimator = PoseEstimator(EstimatorMode.ENCODER_IMU_EXTERNAL)
        estimator.process_encoder(encoder(0, 4))
        for index in range(1, 20):
            t = index * .05
            estimator.process_imu(ImuYawRateMeasurement(t - .01, .01))
            estimator.process_encoder(encoder(t, 4.01, 3.99))
        return estimator.pose, estimator.covariance
    first, second = run(), run()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])


def test_external_fusion_reduces_slip_error_in_controller_loop():
    points = np.column_stack((np.zeros(41), np.linspace(0, 2, 41)))
    samples = tuple(TrajectorySample(float(x), float(y), 0, 0, 2, False,
                                     0 if i == 40 else .4, 0, i * .125, i * .05)
                    for i, (x, y) in enumerate(points))
    generated = GeneratedTrajectory(points, points, points, samples, False, False, (), {})
    command = TrajectoryCommand(1, 1, generated)
    vehicle = VehicleConfig(left_motor_scale=.94, right_motor_scale=1.02,
                            longitudinal_slip=.08, encoder_noise_std_rad_s=.03)
    sensors = SensorSimulationConfig(imu_bias_rps=.002, external_interval_s=.4)
    config = EstimatorConfig(encoder_linear_noise_mps=.1)
    encoder_result = simulate_fused_trajectory(command, vehicle, EstimatorMode.ENCODER_ONLY,
                                                estimator_config=config, sensors=sensors,
                                                seed=5, max_time_s=15)
    fused_result = simulate_fused_trajectory(command, vehicle,
                                              EstimatorMode.ENCODER_IMU_EXTERNAL,
                                              estimator_config=config, sensors=sensors,
                                              seed=5, max_time_s=15)
    assert fused_result.metrics.final_position_error_m < encoder_result.metrics.final_position_error_m
    assert fused_result.metrics.controller_state.value == "stopped"


def test_delayed_imu_and_dropouts_are_deterministic():
    points = np.array([[0., 0.], [0., .5], [0., 1.]])
    samples = tuple(TrajectorySample(x, y, 0, 0, 2, False, 0 if i == 2 else .3,
                                     0, i, y) for i, (x, y) in enumerate(points))
    command = TrajectoryCommand(1, 1, GeneratedTrajectory(points, points, points, samples,
                                                           False, False, (), {}))
    sensors = SensorSimulationConfig(imu_delay_s=.08, encoder_dropout=(.3, .6),
                                     imu_dropout=(.7, .9))
    first = simulate_fused_trajectory(command, VehicleConfig(), EstimatorMode.ENCODER_IMU,
                                      sensors=sensors, seed=9, max_time_s=8)
    second = simulate_fused_trajectory(command, VehicleConfig(), EstimatorMode.ENCODER_IMU,
                                       sensors=sensors, seed=9, max_time_s=8)
    assert first.metrics == second.metrics
    assert any(first.encoder_dropout_mask) and any(first.imu_dropout_mask)
