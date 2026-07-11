"""Rotating range sensor geometry."""

import numpy as np
from numpy.typing import NDArray

from .frames import RigidTransform


def ranges_to_sensor_points(
    ranges_m: NDArray[np.floating],
    azimuth_rad: NDArray[np.floating],
    elevation_rad: NDArray[np.floating] | float = 0.0,
    *,
    min_range_m: float = 0.05,
    max_range_m: float = np.inf,
) -> NDArray[np.float64]:
    """Convert spherical measurements into the ToF sensor's Cartesian frame.

    Sensor axes follow vehicle-style conventions: x right, y forward, z up.
    Azimuth zero is +y and increases toward +x; elevation is positive upward.
    Invalid/out-of-range samples are omitted.
    """
    ranges, azimuth, elevation = np.broadcast_arrays(
        np.asarray(ranges_m, dtype=np.float64),
        np.asarray(azimuth_rad, dtype=np.float64),
        np.asarray(elevation_rad, dtype=np.float64),
    )
    if not np.isfinite(min_range_m) or min_range_m < 0:
        raise ValueError("min_range_m must be finite and non-negative")
    if np.isnan(max_range_m) or max_range_m <= min_range_m:
        raise ValueError("max_range_m must exceed min_range_m")
    valid = (
        np.isfinite(ranges) & np.isfinite(azimuth) & np.isfinite(elevation)
        & (ranges >= min_range_m) & (ranges <= max_range_m)
    )
    r, az, el = ranges[valid], azimuth[valid], elevation[valid]
    horizontal = r * np.cos(el)
    return np.column_stack(
        (horizontal * np.sin(az), horizontal * np.cos(az), r * np.sin(el))
    )


def ranges_to_vehicle_points(
    ranges_m: NDArray[np.floating],
    azimuth_rad: NDArray[np.floating],
    elevation_rad: NDArray[np.floating] | float = 0.0,
    *,
    sensor_to_vehicle: RigidTransform = RigidTransform(),
    min_range_m: float = 0.05,
    max_range_m: float = np.inf,
) -> NDArray[np.float64]:
    """Convert ranges to vehicle coordinates using the sensor mounting pose."""
    return sensor_to_vehicle.apply(ranges_to_sensor_points(
        ranges_m, azimuth_rad, elevation_rad,
        min_range_m=min_range_m, max_range_m=max_range_m,
    ))

