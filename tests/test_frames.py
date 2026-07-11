import numpy as np
import pytest

from carvision.frames import CAMERA_OPTICAL_TO_VEHICLE, RigidTransform, camera_to_vehicle


def rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_known_rotation_and_translation_uses_column_vector_convention() -> None:
    transform = RigidTransform(rotation_z(np.pi / 2), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(transform.apply([2.0, 0.0, 1.0]), [1.0, 4.0, 4.0], atol=1e-12)


def test_transform_inverse_round_trip_random_points() -> None:
    rng = np.random.default_rng(42)
    transform = RigidTransform(rotation_z(0.73), np.array([-0.4, 1.2, 0.8]))
    points = rng.normal(size=(100, 3))
    np.testing.assert_allclose(transform.inverse().apply(transform.apply(points)), points, atol=1e-12)


def test_camera_mounted_above_ahead_and_yawed() -> None:
    # Full optical-camera-to-vehicle rotation: canonical axes, then 90-degree yaw.
    rotation = rotation_z(np.pi / 2) @ CAMERA_OPTICAL_TO_VEHICLE.rotation
    mount = RigidTransform(rotation, np.array([0.0, 0.5, 1.2]))
    # Optical center maps to mount translation; optical-forward maps vehicle-left after yaw.
    np.testing.assert_allclose(camera_to_vehicle([[0, 0, 0], [0, 0, 2]], mount),
                               [[0, 0.5, 1.2], [-2, 0.5, 1.2]], atol=1e-12)


def test_invalid_rotation_is_rejected() -> None:
    with pytest.raises(ValueError):
        RigidTransform(np.diag([1.0, 1.0, 2.0]), np.zeros(3))
    with pytest.raises(ValueError):
        RigidTransform(np.diag([1.0, 1.0, -1.0]), np.zeros(3))

