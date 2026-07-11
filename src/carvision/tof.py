"""Rotating range sensor geometry."""

import numpy as np
from numpy.typing import NDArray


def ranges_to_vehicle_points(
    ranges_m: NDArray[np.floating],
    azimuth_rad: NDArray[np.floating],
    elevation_rad: NDArray[np.floating] | float = 0.0,
    *,
    min_range_m: float = 0.05,
    max_range_m: float = np.inf,
) -> NDArray[np.float64]:
    """Convert matched spherical ranges into vehicle-frame points.

    Azimuth zero points forward (+y) and increases toward +x. Elevation is
    positive upward. Invalid/out-of-range samples are omitted.
    """
    ranges, azimuth, elevation = np.broadcast_arrays(
        np.asarray(ranges_m, dtype=np.float64),
        np.asarray(azimuth_rad, dtype=np.float64),
        np.asarray(elevation_rad, dtype=np.float64),
    )
    if min_range_m < 0 or max_range_m <= min_range_m:
        raise ValueError("range limits must satisfy 0 <= min < max")
    valid = (
        np.isfinite(ranges)
        & np.isfinite(azimuth)
        & np.isfinite(elevation)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
    )
    r, az, el = ranges[valid], azimuth[valid], elevation[valid]
    horizontal = r * np.cos(el)
    return np.column_stack(
        (horizontal * np.sin(az), horizontal * np.cos(az), r * np.sin(el))
    )

