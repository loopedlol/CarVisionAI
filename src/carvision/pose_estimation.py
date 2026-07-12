"""Timestamped, inspectable planar pose estimation for differential drive.

State is ``[x_right_m, y_forward_m, heading_rad, imu_yaw_bias_rps]``.
Encoder velocity events propagate pose and covariance. In IMU modes, the most
recent yaw-rate observation (minus estimated bias) supplies angular velocity;
encoder angular velocity remains the fallback and bias reference. External
heading/pose observations are ordinary gated Kalman updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, sin
import numpy as np
from numpy.typing import NDArray

from .control import Pose2D, WheelSpeeds, wheels_to_body, wrap_angle


@dataclass(frozen=True)
class EncoderMeasurement:
    timestamp_s: float
    wheels_rad_s: WheelSpeeds


@dataclass(frozen=True)
class ImuYawRateMeasurement:
    timestamp_s: float
    yaw_rate_rps: float
    variance_rps2: float | None = None


@dataclass(frozen=True)
class HeadingMeasurement:
    timestamp_s: float
    heading_rad: float
    variance_rad2: float | None = None


@dataclass(frozen=True)
class ExternalPoseMeasurement:
    timestamp_s: float
    pose: Pose2D
    covariance: NDArray[np.float64] | None = None


class EstimatorMode(str, Enum):
    ENCODER_ONLY = "encoder_only"
    ENCODER_IMU = "encoder_imu"
    ENCODER_IMU_EXTERNAL = "encoder_imu_external"


class EstimatorHealth(str, Enum):
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"


class EstimatorFault(str, Enum):
    NONE = "none"
    TIMESTAMP_DISORDER = "timestamp_disorder"
    NONFINITE_MEASUREMENT = "nonfinite_measurement"
    UNCERTAINTY_EXCESSIVE = "uncertainty_excessive"


@dataclass(frozen=True)
class EstimatorConfig:
    wheel_radius_m: float = .065
    track_width_m: float = .32
    encoder_linear_noise_mps: float = .025
    encoder_angular_noise_rps: float = .08
    gyro_noise_rps: float = .025
    gyro_bias_random_walk_rps_sqrt_s: float = .004
    imu_yaw_rate_weight: float = .90
    heading_noise_rad: float = .06
    external_position_noise_m: float = .08
    external_heading_noise_rad: float = .08
    initial_position_std_m: float = .02
    initial_heading_std_rad: float = .03
    initial_bias_std_rps: float = .08
    gate_chi2_heading: float = 9.0
    gate_chi2_pose: float = 11.34
    max_delayed_measurement_s: float = .20
    imu_timeout_s: float = .20
    degraded_position_std_m: float = .45
    fault_position_std_m: float = 1.2
    degraded_heading_std_rad: float = .45

    def __post_init__(self) -> None:
        values = vars(self)
        if any(not np.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError("all estimator configuration values must be finite and positive")
        if self.imu_yaw_rate_weight > 1:
            raise ValueError("imu_yaw_rate_weight must not exceed one")


@dataclass(frozen=True)
class EstimatorStatus:
    timestamp_s: float | None
    pose: Pose2D
    covariance: NDArray[np.float64]
    gyro_bias_rps: float
    health: EstimatorHealth
    fault: EstimatorFault
    accepted_encoder: int
    accepted_imu: int
    accepted_heading: int
    accepted_external: int
    rejected_outliers: int
    rejected_timestamps: int
    imu_stale: bool


class PoseEstimator:
    """Small EKF-like estimator with direct wheel propagation and gated updates."""

    def __init__(self, mode: EstimatorMode, config: EstimatorConfig = EstimatorConfig()) -> None:
        self.mode, self.config = mode, config
        self.reset(Pose2D(0, 0, 0), 0.0)

    def reset(self, pose: Pose2D, timestamp_s: float = 0.0) -> None:
        if not _finite(timestamp_s, pose.x_m, pose.y_m, pose.heading_rad):
            raise ValueError("reset pose and timestamp must be finite")
        c = self.config
        self.state = np.array([pose.x_m, pose.y_m, wrap_angle(pose.heading_rad), 0.0])
        self.covariance = np.diag([c.initial_position_std_m ** 2] * 2
                                  + [c.initial_heading_std_rad ** 2,
                                     c.initial_bias_std_rps ** 2])
        self.timestamp_s: float | None = timestamp_s
        self.last_encoder_timestamp_s: float | None = None
        self.last_imu_timestamp_s: float | None = None
        self.last_heading_timestamp_s: float | None = None
        self.last_external_timestamp_s: float | None = None
        self.latest_imu_rate_rps: float | None = None
        self.last_encoder_omega_rps = 0.0
        self.health, self.fault = EstimatorHealth.INITIALIZING, EstimatorFault.NONE
        self.counts = {"encoder": 0, "imu": 0, "heading": 0, "external": 0,
                       "outlier": 0, "timestamp": 0}

    @property
    def pose(self) -> Pose2D:
        return Pose2D(float(self.state[0]), float(self.state[1]), float(self.state[2]))

    def process_encoder(self, measurement: EncoderMeasurement) -> bool:
        if not _finite(measurement.timestamp_s, measurement.wheels_rad_s.left_rad_s,
                       measurement.wheels_rad_s.right_rad_s):
            return self._invalid()
        if not self._timestamp_ok(measurement.timestamp_s, self.last_encoder_timestamp_s,
                                  allow_delayed=False):
            return False
        if self.last_encoder_timestamp_s is None:
            self.last_encoder_timestamp_s = measurement.timestamp_s
            self.timestamp_s = max(self.timestamp_s or measurement.timestamp_s,
                                   measurement.timestamp_s)
            self.counts["encoder"] += 1
            self._update_health()
            return True
        dt = measurement.timestamp_s - self.last_encoder_timestamp_s
        self.last_encoder_timestamp_s = measurement.timestamp_s
        self.timestamp_s = max(self.timestamp_s or measurement.timestamp_s,
                               measurement.timestamp_s)
        v, encoder_omega = wheels_to_body(measurement.wheels_rad_s,
                                          self.config.wheel_radius_m,
                                          self.config.track_width_m)
        self.last_encoder_omega_rps = encoder_omega
        imu_fresh = (self.latest_imu_rate_rps is not None and self.last_imu_timestamp_s is not None
                     and measurement.timestamp_s - self.last_imu_timestamp_s
                     <= self.config.imu_timeout_s)
        if self.mode != EstimatorMode.ENCODER_ONLY and imu_fresh:
            imu_omega = self.latest_imu_rate_rps - self.state[3]
            weight = self.config.imu_yaw_rate_weight
            omega = weight * imu_omega + (1 - weight) * encoder_omega
        else:
            omega = encoder_omega
        self._predict(v, float(omega), dt, imu_fresh)
        self.counts["encoder"] += 1
        self._update_health()
        return True

    def process_imu(self, measurement: ImuYawRateMeasurement) -> bool:
        if self.mode == EstimatorMode.ENCODER_ONLY:
            return False
        if not _finite(measurement.timestamp_s, measurement.yaw_rate_rps):
            return self._invalid()
        if not self._timestamp_ok(measurement.timestamp_s, self.last_imu_timestamp_s,
                                  allow_delayed=True):
            return False
        self.advance_time(measurement.timestamp_s)
        # A yaw-rate sample alone cannot distinguish true rotation from gyro
        # bias. Store it (with bounded-delay handling); heading/pose corrections
        # make bias observable through prediction cross-covariance.
        self.latest_imu_rate_rps = measurement.yaw_rate_rps
        self.last_imu_timestamp_s = measurement.timestamp_s
        self.counts["imu"] += 1
        self._update_health()
        return True

    def note_encoder_dropout(self, timestamp_s: float) -> bool:
        """Advance uncertainty without integrating a future sample over the gap."""
        if self.last_encoder_timestamp_s is not None and timestamp_s <= self.last_encoder_timestamp_s:
            return False
        self.last_encoder_timestamp_s = timestamp_s
        return self.advance_time(timestamp_s)

    def advance_time(self, timestamp_s: float) -> bool:
        """Grow uncertainty during a period with no wheel propagation."""
        if not np.isfinite(timestamp_s) or (self.timestamp_s is not None
                                             and timestamp_s < self.timestamp_s):
            return False
        dt = 0.0 if self.timestamp_s is None else timestamp_s - self.timestamp_s
        self.timestamp_s = timestamp_s
        if dt > 0:
            c = self.config
            self.covariance += np.diag([
                (c.encoder_linear_noise_mps * dt) ** 2 + 1e-4 * dt,
                (c.encoder_linear_noise_mps * dt) ** 2 + 1e-4 * dt,
                (c.encoder_angular_noise_rps * dt) ** 2,
                c.gyro_bias_random_walk_rps_sqrt_s ** 2 * dt,
            ])
            self._update_health()
        return True

    def process_heading(self, measurement: HeadingMeasurement) -> bool:
        if self.mode == EstimatorMode.ENCODER_ONLY:
            return False
        if not _finite(measurement.timestamp_s, measurement.heading_rad):
            return self._invalid()
        if not self._timestamp_ok(measurement.timestamp_s, self.last_heading_timestamp_s,
                                  allow_delayed=True):
            return False
        delay = max(0.0, (self.timestamp_s or measurement.timestamp_s) - measurement.timestamp_s)
        innovation = wrap_angle(measurement.heading_rad - self.state[2])
        h = np.array([[0., 0., 1., 0.]])
        accepted = self._scalar_update(innovation, h,
                                       (measurement.variance_rad2 or self.config.heading_noise_rad ** 2)
                                       + (delay * self.config.gyro_noise_rps) ** 2,
                                       self.config.gate_chi2_heading)
        self.last_heading_timestamp_s = measurement.timestamp_s
        if accepted: self.counts["heading"] += 1
        self._update_health()
        return accepted

    def process_external_pose(self, measurement: ExternalPoseMeasurement) -> bool:
        if self.mode != EstimatorMode.ENCODER_IMU_EXTERNAL:
            return False
        pose = measurement.pose
        if not _finite(measurement.timestamp_s, pose.x_m, pose.y_m, pose.heading_rad):
            return self._invalid()
        if not self._timestamp_ok(measurement.timestamp_s, self.last_external_timestamp_s,
                                  allow_delayed=True):
            return False
        delay = max(0.0, (self.timestamp_s or measurement.timestamp_s) - measurement.timestamp_s)
        default = np.diag([self.config.external_position_noise_m ** 2] * 2
                          + [self.config.external_heading_noise_rad ** 2])
        r = default if measurement.covariance is None else np.asarray(measurement.covariance, float)
        if r.shape != (3, 3) or not np.isfinite(r).all() or np.min(np.linalg.eigvalsh(r)) <= 0:
            raise ValueError("external covariance must be finite positive-definite 3x3")
        r = r + np.diag([(delay * self.config.encoder_linear_noise_mps) ** 2] * 2
                        + [(delay * self.config.gyro_noise_rps) ** 2])
        h = np.zeros((3, 4)); h[:3, :3] = np.eye(3)
        innovation = np.array([pose.x_m - self.state[0], pose.y_m - self.state[1],
                               wrap_angle(pose.heading_rad - self.state[2])])
        s = h @ self.covariance @ h.T + r
        mahalanobis = float(innovation @ np.linalg.solve(s, innovation))
        if mahalanobis > self.config.gate_chi2_pose:
            self.counts["outlier"] += 1
            self.last_external_timestamp_s = measurement.timestamp_s
            return False
        gain = self.covariance @ h.T @ np.linalg.inv(s)
        self.state += gain @ innovation
        self.state[2] = wrap_angle(self.state[2])
        self._joseph(gain, h, r)
        self.counts["external"] += 1
        self.last_external_timestamp_s = measurement.timestamp_s
        self._update_health()
        return True

    def status(self) -> EstimatorStatus:
        imu_stale = (self.mode != EstimatorMode.ENCODER_ONLY
                     and (self.last_imu_timestamp_s is None or self.timestamp_s is None
                          or self.timestamp_s - self.last_imu_timestamp_s > self.config.imu_timeout_s))
        return EstimatorStatus(self.timestamp_s, self.pose, self.covariance.copy(),
                               float(self.state[3]), self.health, self.fault,
                               self.counts["encoder"], self.counts["imu"],
                               self.counts["heading"], self.counts["external"],
                               self.counts["outlier"], self.counts["timestamp"], imu_stale)

    def _predict(self, velocity: float, omega: float, dt: float, imu_used: bool) -> None:
        if dt <= 0: return
        heading = self.state[2]
        midpoint = heading + omega * dt / 2
        self.state[0] += velocity * sin(midpoint) * dt
        self.state[1] += velocity * cos(midpoint) * dt
        self.state[2] = wrap_angle(heading + omega * dt)
        f = np.eye(4)
        f[0, 2] = velocity * cos(midpoint) * dt
        f[1, 2] = -velocity * sin(midpoint) * dt
        if imu_used:
            f[0, 3] = -velocity * cos(midpoint) * dt ** 2 / 2
            f[1, 3] = velocity * sin(midpoint) * dt ** 2 / 2
            f[2, 3] = -dt
        q_v = self.config.encoder_linear_noise_mps ** 2
        q_w = (self.config.gyro_noise_rps if imu_used
               else self.config.encoder_angular_noise_rps) ** 2
        g = np.array([[sin(midpoint) * dt, 0.], [cos(midpoint) * dt, 0.],
                      [0., dt], [0., 0.]])
        q = g @ np.diag([q_v, q_w]) @ g.T
        q[3, 3] += self.config.gyro_bias_random_walk_rps_sqrt_s ** 2 * dt
        self.covariance = f @ self.covariance @ f.T + q
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _scalar_update(self, innovation: float, h: NDArray[np.float64], variance: float,
                       gate: float) -> bool:
        s = float((h @ self.covariance @ h.T)[0, 0] + variance)
        if innovation ** 2 / s > gate:
            self.counts["outlier"] += 1
            return False
        gain = self.covariance @ h.T / s
        self.state += gain[:, 0] * innovation
        self.state[2] = wrap_angle(self.state[2])
        self._joseph(gain, h, np.array([[variance]]))
        return True

    def _joseph(self, gain: NDArray, h: NDArray, r: NDArray) -> None:
        identity = np.eye(4)
        residual = identity - gain @ h
        self.covariance = residual @ self.covariance @ residual.T + gain @ r @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2

    def _timestamp_ok(self, timestamp: float, sensor_last: float | None, *,
                      allow_delayed: bool) -> bool:
        if sensor_last is not None and timestamp <= sensor_last:
            self.counts["timestamp"] += 1; self.fault = EstimatorFault.TIMESTAMP_DISORDER
            return False
        if (self.timestamp_s is not None and timestamp < self.timestamp_s
                and (not allow_delayed or self.timestamp_s - timestamp
                     > self.config.max_delayed_measurement_s)):
            self.counts["timestamp"] += 1; self.fault = EstimatorFault.TIMESTAMP_DISORDER
            return False
        return True

    def _invalid(self) -> bool:
        self.fault = EstimatorFault.NONFINITE_MEASUREMENT
        return False

    def _update_health(self) -> None:
        position_std = float(np.sqrt(max(self.covariance[0, 0], self.covariance[1, 1])))
        heading_std = float(np.sqrt(self.covariance[2, 2]))
        if position_std >= self.config.fault_position_std_m:
            self.health, self.fault = EstimatorHealth.FAULT, EstimatorFault.UNCERTAINTY_EXCESSIVE
        elif (position_std >= self.config.degraded_position_std_m
              or heading_std >= self.config.degraded_heading_std_rad):
            self.health = EstimatorHealth.DEGRADED
        else:
            self.health = EstimatorHealth.HEALTHY
            if self.fault == EstimatorFault.UNCERTAINTY_EXCESSIVE:
                self.fault = EstimatorFault.NONE


def _finite(*values: float) -> bool:
    return bool(np.isfinite(values).all())
