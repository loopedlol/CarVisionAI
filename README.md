# CarVisionAI perception prototype

This repository contains a small, sensor-independent perception pipeline that
can be exercised without hardware. It intentionally stops at a local 2D
occupancy grid; localization, tracking, planning, and vehicle control are out of
scope.

## Coordinate frames and units

All distances are metres and angles are radians.

- **Image/camera frame:** `x` right, `y` down, `z` forward. Pixels use `(u, v)`
  = (column, row). Depth is optical-axis `z`, not Euclidean range.
- **Vehicle/local mapping frame:** `x` right, `y` forward, `z` up. Camera points
  convert as `(x_v, y_v, z_v) = (x_c, z_c, -y_c)`.
- **Occupancy grid:** array rows increase with vehicle `y`; columns increase
  with vehicle `x`. Cell `(row, column)` covers half-open metric bounds. The
  grid stores `-1` unknown, `0` free, and `100` occupied.

The mapper accepts only vehicle-frame points and a sensor origin, so stereo and
ToF processing do not leak into mapping.

## Setup and use

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/python scripts/run_stereo_stages.py --output-dir outputs/stereo
.venv/bin/python scripts/run_tof_pipeline.py --output-dir outputs/tof
.venv/bin/python scripts/run_mock_pipeline.py --sensor both --output-dir outputs/combined
```

Each script writes `.npy` arrays and PNG previews. White is occupied, dark gray
is free, and mid-gray is unknown in occupancy previews.

