"""Rigid coordinate-frame transforms using column-vector geometry."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RigidTransform:
    """A source-to-target rigid transform: ``p_target = R @ p_source + t``."""

    rotation: NDArray[np.float64] = field(default_factory=lambda: np.eye(3))
    translation_m: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation_m, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("rotation must be 3x3 and translation_m must have shape (3,)")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("transform values must be finite")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9):
            raise ValueError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError("rotation must be proper (determinant +1)")
        object.__setattr__(self, "rotation", rotation.copy())
        object.__setattr__(self, "translation_m", translation.copy())

    def apply(self, points_source: NDArray[np.floating]) -> NDArray[np.float64]:
        """Transform one point or an array shaped (..., 3); NaNs propagate."""
        points = np.asarray(points_source, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("points_source must have shape (..., 3)")
        return points @ self.rotation.T + self.translation_m

    def inverse(self) -> "RigidTransform":
        rotation = self.rotation.T
        return RigidTransform(rotation, -(rotation @ self.translation_m))


# Canonical optical camera fixed at the vehicle origin, looking forward.
CAMERA_OPTICAL_TO_VEHICLE = RigidTransform(
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
)


def camera_to_vehicle(
    points_camera: NDArray[np.floating],
    camera_to_vehicle_transform: RigidTransform = CAMERA_OPTICAL_TO_VEHICLE,
) -> NDArray[np.float64]:
    """Transform optical-camera points (x right, y down, z forward) to vehicle."""
    return camera_to_vehicle_transform.apply(points_camera)

