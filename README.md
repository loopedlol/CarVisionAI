# CarVisionAI

A modular autonomy stack for a small differential-drive vehicle, built to explore the path from **perception to local navigation and control**. The project now spans stereo/ToF-style perception, occupancy mapping, route planning, event-driven replanning, trajectory generation, closed-loop control, pose estimation, offline sensor analysis, and optional 3D visualization.

The architecture deliberately keeps sensor-specific processing separate from mapping and downstream autonomy, so most of the stack can be exercised with synthetic or recorded data before hardware is connected.

**Stack:** Python · NumPy · OpenCV · pytest · optional Open3D

> **Project status:** this is a development and simulation toolkit, not a production autonomous-driving or safety system. Several components have recorded-data paths, but synthetic tests and simplified models are still used throughout the stack.

## System overview

```text
Stereo / ToF / recorded sensor data
              ↓
     Metric point generation
              ↓
      Ternary occupancy map
              ↓
 Temporal map-state filtering
              ↓
 Multi-candidate local planning
              ↓
    Trajectory generation
              ↓
 Closed-loop differential-drive control
              ↓
 Pose estimation + offline tuning
              ↓
    Optional 3D visualization
```

### Key capabilities

- stereo calibration, rectification, disparity filtering, and depth projection;
- sensor-independent local occupancy mapping with explicit coordinate transforms;
- A* local planning with clearance, unknown-space, and heading-change costs;
- temporal map-change detection and selective replanning rather than unconditional full replans;
- conservative path-to-trajectory generation with collision checks and speed limits;
- simulated pure-pursuit control, watchdog/fault handling, and differential-drive commands;
- encoder/IMU pose estimation plus recorded-data replay, calibration, and bounded tuning;
- optional Open3D snapshots and replay without making visualization a runtime dependency.

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

Sensor mounting uses `RigidTransform` with the explicit convention
`p_vehicle = R_sensor_to_vehicle @ p_sensor + t_sensor_in_vehicle`. Supply its
translation as the occupancy update's sensor origin when ray-clearing from an
offset sensor. For cameras, the supplied rotation is the full optical-frame to
vehicle-frame rotation; the default `CAMERA_OPTICAL_TO_VEHICLE` constant covers
only a camera at the vehicle origin looking straight ahead.

## Setup and use

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/python scripts/run_stereo_stages.py --output-dir outputs/stereo
.venv/bin/python scripts/run_tof_pipeline.py --output-dir outputs/tof
.venv/bin/python scripts/run_mock_pipeline.py --sensor both --output-dir outputs/combined
.venv/bin/python scripts/run_noisy_pipeline.py --output-dir outputs/noisy
.venv/bin/python scripts/benchmark_dense.py
```

Each script writes `.npy` arrays and PNG previews. White is occupied, dark gray
is free, and mid-gray is unknown in occupancy previews.

## Recorded stereo adapter

The recorded adapter preserves the core pipeline boundary: OpenCV produces a
filtered floating-point disparity image, then `stereo.py`, `RigidTransform`, and
`OccupancyGrid` perform geometry and mapping.

Calibration is JSON; see `config/example_stereo_calibration.json`. It records:

- `image_size` as `[width, height]`;
- left and right OpenCV camera matrices and distortion vectors;
- `right_from_left` using OpenCV's convention `p_right = R @ p_left + T`, with
  `T` in metres (a conventional horizontal rig normally has negative `T.x`);
- the full left optical-camera to vehicle transform, also in metres;
- optional RMS reprojection error metadata.

Calibration from matched checkerboard folders (filenames must match):

```bash
.venv/bin/python scripts/calibrate_stereo.py \
  --left-dir data/calibration/left --right-dir data/calibration/right \
  --columns 9 --rows 6 --square-size-m 0.025 \
  --camera-translation-m 0 0.25 0.8 \
  --output config/my_stereo_rig.json
```

`columns` and `rows` are inner-corner counts. Camera roll/pitch/yaw can be
supplied with `--camera-rpy-deg`; they are applied in vehicle axes after the
canonical optical-camera axis conversion.

Inspect rectification and process data:

```bash
.venv/bin/python scripts/inspect_rectification.py \
  --calibration config/my_stereo_rig.json --left frame_left.png \
  --right frame_right.png --output outputs/rectification

.venv/bin/python scripts/process_stereo_pair.py \
  --calibration config/my_stereo_rig.json --left frame_left.png \
  --right frame_right.png --output-dir outputs/recorded_pair

.venv/bin/python scripts/process_stereo_folder.py \
  --calibration config/my_stereo_rig.json --left-dir data/left \
  --right-dir data/right --output-dir outputs/sequence --save-every 10

.venv/bin/python scripts/benchmark_recorded_pipeline.py \
  --calibration config/my_stereo_rig.json --left frame_left.png \
  --right frame_right.png --iterations 20
```

Useful matcher controls are exposed directly: `--min-disparity`,
`--num-disparities` (multiple of 16), `--block-size`, `--uniqueness-ratio`,
`--speckle-window-size`, `--lr-max-diff`, and metric depth limits. Negative
`--lr-max-diff` disables the relatively expensive right-to-left consistency
check. OpenCV's signed fixed-point disparity output is divided by 16 before any
geometry. Invalid, inconsistent, out-of-ROI, and out-of-depth-range pixels are
stored as NaN in the filtered disparity and depth arrays.

## Recorded-data evaluation

An evaluation dataset is a directory containing a versioned `dataset.json`, its
calibration, and images. Paths must stay below the manifest directory. A compact
manifest looks like:

```json
{
  "format_version": 1,
  "name": "first_rig_evaluation",
  "calibration": "calibration.json",
  "expected_image_size": [640, 480],
  "default_sgbm": {"num_disparities": 128, "max_depth_m": 15.0},
  "notes": "Rigid tripod sequence",
  "scenes": [{
    "id": "indoor_wall_2m",
    "category": "static_distance_target",
    "static": true,
    "conditions": {"lighting": "dim", "texture": "low", "motion": "none"},
    "notes": "Laser-measured perpendicular distance",
    "frames": [{
      "id": "0000",
      "left": "left/0000.png",
      "right": "right/0000.png",
      "regions": [{
        "name": "wall_center",
        "rect": [260, 180, 120, 100],
        "expected_depth_m": 2.0,
        "tags": ["textureless"],
        "notes": "Avoid the target edges"
      }]
    }]
  }]
}
```

ROI rectangles are `[x, y, width, height]` in the rectified left image. Measured
depth is optical-axis depth, not slant range. Frames in a `static: true` scene
are used for temporal depth and consecutive occupancy stability.

Run one frame, a dataset, a bounded parameter comparison, and report rendering:

```bash
.venv/bin/python scripts/evaluate_stereo_pair.py \
  --dataset data/eval/dataset.json --scene indoor_wall_2m --frame 0000 \
  --output-dir outputs/eval_pair

.venv/bin/python scripts/evaluate_stereo_dataset.py \
  --dataset data/eval/dataset.json --output-dir outputs/eval_dataset

.venv/bin/python scripts/compare_stereo_configs.py \
  --dataset data/eval/dataset.json --configs config/example_evaluation_configs.json \
  --output-dir outputs/eval_comparison

.venv/bin/python scripts/generate_evaluation_report.py \
  --metrics outputs/eval_dataset/metrics.json --output outputs/eval_dataset/report.md
```

Comparison JSON explicitly names at most 12 partial `SGBMConfig` overrides. The
recommended example compares baseline, faster matching without the second
left-right pass, and a stricter robust configuration. Every configuration sees
the identical manifest frames and ROIs. Results remain a quality/runtime table;
the toolkit deliberately does not collapse tradeoffs into one opaque score.

## Local navigation and directed inspection

The first planning layer consumes only the ternary `OccupancyGrid` contract; it
does not import stereo or ToF code. Vehicle-frame metric poses can be supplied
with `PlanningPose` (`heading=0` means forward `+y`, positive turns toward
vehicle `+x`), or callers can use `(row, column)` cells directly.

Planning consists of:

- Euclidean obstacle distance and footprint inflation;
- configurable unknown handling: `block`, `penalize`, or `allow`;
- 8-connected A* with corner-cut prevention, clearance, narrow-passage,
  unknown, and heading-change edge costs;
- repeated corridor penalties plus overlap rejection for meaningfully distinct
  candidate routes;
- transparent rescoring into length, clearance, unknown distance, narrow
  distance, heading change, traversal difficulty, and total;
- inspection targeting for unknown route cells, route divergence, narrow
  passages, and isolated ToF-style returns;
- circular mock stereo refinement from a separate ground-truth map followed by
  ordinary replanning.

Run the complete synthetic active-perception loop and planning benchmark:

```bash
.venv/bin/python scripts/run_active_perception_demo.py \
  --output-dir outputs/planning_demo
.venv/bin/python scripts/benchmark_planning.py
```

The demo saves rough candidates, inspection targeting, changed cells, the
replanned candidates, a combined sequence image, and machine-readable score
components. The circular inspection region is a bounded mock observation, not
a physical stereo frustum model.

## Event-driven map changes and replanning

`TemporalMapState` sits between observed ternary occupancy frames and planning.
It stores stable state, confidence, observation count, a pending transition and
its persistence count, and the last raw observation for every cell. Defaults
require two occupied observations, three free observations, or three unknown
observations before changing stable state. Directed observations can provide a
larger integer evidence weight but use the same transition machine. Confidence
below the configured minimum cannot advance a transition.

The trigger policy builds masks for the selected path, exact footprint,
inflated safety corridor, stopping-distance path prefix, all candidates,
candidate divergence, active inspection regions, and the goal. Stable changes
are weighted by their strongest navigation relationship; distant cells carry
negligible aggregate weight. Raw occupied evidence in the stopping corridor and
widespread raw corruption bypass persistence and request an immediate stop.

Actions are explicit:

- `no_action`;
- `update_route_metrics`;
- `reevaluate_candidates`;
- `fast_single_path_replan`;
- `full_multi_candidate_replan`;
- `immediate_stop_invalid_route`.

Run the continuous synthetic timeline and its always-full-planning benchmark:

```bash
.venv/bin/python scripts/run_map_event_simulation.py \
  --output-dir outputs/map_events
```

The command saves one annotated image per frame, a contact-sheet timeline, and
JSON containing raw/stable/relevant changes, action and reason, route validity,
policy planning time, and the cost of full five-candidate replanning on the same
observation. This is a trigger-policy simulation, not a substitute for an
independent emergency braking system.

## Conservative trajectory generation

The trajectory layer consumes one selected grid path. It converts cell centers
to metres, removes duplicates and collinear points, greedily keeps the farthest
collision-visible waypoint, and optionally rounds corners with sampled
quadratic Bézier segments. Every shortcut and curve is sampled more densely
than half a grid cell and checked against obstacle clearance for the circular
robot footprint plus safety margin. Unsafe corner rounding falls back to the
simplified or raw grid path.

The sampled differential-drive reference contains timestamp, vehicle-frame
`x/y`, heading (zero is `+y`), curvature, clearance, unknown flag, target linear
velocity, and target angular velocity. Speed limits combine curvature, angular
velocity, clearance, unknown exposure, remaining goal distance, and configured
maximum velocity. Forward acceleration and backward braking passes enforce
linear limits and guarantee zero terminal velocity. Optional final heading is
implemented as an in-place rotation followed by a zero command.

Stopping distance is conservative and includes reaction latency:

```text
d_stop = speed * reaction_latency
       + speed² / (2 * maximum_deceleration)
       + braking_safety_margin
```

Run the demonstration and benchmark:

```bash
.venv/bin/python scripts/run_trajectory_demo.py \
  --output-dir outputs/trajectory_demo
.venv/bin/python scripts/benchmark_trajectory.py
```

Updated stable maps produce one of: trajectory valid, reduce speed, regenerate
from the same selected path, fast path replan, or immediate stop. Event-policy
stop and path-replanning decisions take precedence, but a local trajectory
regeneration does not force full multi-candidate planning.

## Simulated closed-loop following

The control layer wraps an existing `GeneratedTrajectory` in a small
`TrajectoryCommand` carrying a monotonically increasing ID and stable-map
revision. A pure-pursuit controller tracks position using pose feedback, then
switches to explicit in-place terminal heading alignment. It reports progress,
commands, wheel speeds, errors, acceptance state, watchdog state, saturation,
and a concrete fault code. Cancel, map invalidation, emergency stop, stale IDs,
revision mismatch, excessive tracking error, persistent saturation, and a
terminal timeout all produce zero commands.

Vehicle coordinates remain `x` right, `y` forward, heading zero along `+y`, and
positive heading toward `+x`. Therefore the differential-drive conversion is:

```text
left_rad_s  = (v + omega * track_width / 2) / wheel_radius
right_rad_s = (v - omega * track_width / 2) / wheel_radius
```

The simulator keeps ground truth separate from encoder-integrated odometry and
supports command delay, wheel saturation, motor mismatch, encoder/pose noise,
longitudinal/angular slip, and motor-response scaling. Run the three-condition
diagnostic and isolated controller benchmark with:

```bash
.venv/bin/python scripts/run_closed_loop_simulation.py \
  --output-dir outputs/closed_loop
.venv/bin/python scripts/benchmark_control.py --iterations 20
```

This is a deterministic control-development simulator. Its motor and slip
models are deliberately simple, and wheel odometry cannot detect unobserved
slip without an independent pose sensor.

## Timestamped pose estimation

`PoseEstimator` adds a small planar EKF-style layer without changing the
controller API. Its state is `[x_right, y_forward, heading, gyro_yaw_bias]`.
Timestamped wheel velocities propagate position and covariance. In IMU modes,
the latest yaw rate is blended with encoder yaw rate for propagation; optional
heading and external `[x, y, heading]` observations use Joseph-form covariance
updates and chi-square innovation gates. Gyro bias becomes observable only when
heading or pose corrections are available.

Measurements must be monotonic per sensor. IMU and correction measurements may
arrive within a configured delay window and are applied at the current state
with conservative handling; the estimator deliberately does not claim full
out-of-sequence history replay. Missing encoder periods grow covariance without
integrating the first returning sample across the entire gap. Health becomes
degraded or faulted when reported uncertainty crosses configured limits.

Run the three-mode moderate-slip comparison, severe dropout case, and estimator
microbenchmark:

```bash
.venv/bin/python scripts/run_pose_fusion_demo.py \
  --output-dir outputs/pose_fusion
.venv/bin/python scripts/benchmark_pose_estimator.py --iterations 10000
```

The synthetic external correction represents a future localization source; it
is not evidence that a real camera, GNSS, beacon, or SLAM system will achieve
the configured covariance or update rate.

## Recorded encoder/IMU datasets and offline tuning

Recorded format version 1 uses an immutable `manifest.json` plus asynchronous
`raw.csv`. Every CSV row identifies its clock and may contain encoder counts or
wheel velocity, IMU yaw rate/heading, external reference pose, battery voltage,
controller command, marker, and notes. Empty fields represent missing samples;
each sensor group also has an explicit validity flag. Preprocessing writes a
separate `processed.csv` with normalized timestamps and wheel velocities.

The manifest documents units, clocks, wheel/encoder metadata, vehicle geometry,
sensor and external frames, calibration version, run type, surface, payload,
and optional independently measured distance or heading change. See
`examples/recorded_sensor_dataset/README.md` for the layout and collection
rules.

The bounded tuner evaluates a 3×2×2×2 candidate set for encoder process noise,
gyro noise, external-position covariance, and pose-gating threshold on explicit training runs. It
selects by mean training position RMSE, then evaluates exactly once on disjoint
held-out runs. It does not silently tune wheel geometry from an evaluation
trajectory.

```bash
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py validate DATASET
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py preprocess DATASET
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py analyze-imu STATIONARY_DATASET
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py calibrate CONTROLLED_RUN
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py replay DATASET --mode encoder_imu_external
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py compare DATASET
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py tune \
  --train TRAIN_A TRAIN_B --evaluate HELD_OUT --output tuning.json
PYTHONPATH=src .venv/bin/python scripts/sensor_log_tool.py report DATASET_A DATASET_B \
  --output-dir outputs/recorded_report
```

Time-offset estimation is reported only when streams contain enough shared
dynamic excitation for correlation. Constant stationary or straight signals
correctly return “not identifiable” instead of inventing an offset.

## Optional native 3D viewer

The versioned `VisualizationSnapshot` is the only boundary between autonomy and
visualization. Publishing is a non-blocking insertion into a bounded queue; the
consumer takes the newest snapshot and counts every obsolete frame it drops.
Open3D is imported only by the renderer, so mapping, planning, control, tests,
and headless deployments do not require it. Use `ViewerConfig(enabled=False)`
for a zero-work disabled publisher.

Open3D coordinates are not remapped: red `+X` is vehicle-right, green `+Y` is
vehicle-forward, and blue `+Z` is up. The foreground event-loop option is
preferred on macOS; mock or autonomy producers can run independently and only
publish snapshots.

Open3D currently has no Python 3.14 wheel. Use Python 3.10–3.12:

```bash
python3.12 -m venv .venv-open3d
.venv-open3d/bin/pip install -e '.[visualization]'
PYTHONPATH=src .venv-open3d/bin/python scripts/view_3d.py live-mock \
  --frames 200 --rate 12
```

Viewer keys:

- `Space`: pause/resume
- `R/T/S/V`: perspective/top/side/front view
- `1/2/3`: stereo/ToF/combined points
- `F/P/O/C`: frames/paths/occupancy/confidence filtering
- `M`: fixed-world/vehicle-following camera
- `+/-`: point size
- `K/I`: save current cloud/screenshot
- `N/B/H`: replay next/previous/restart

Other commands:

```bash
PYTHONPATH=src .venv-open3d/bin/python scripts/view_3d.py smoke-test
PYTHONPATH=src .venv-open3d/bin/python scripts/view_3d.py static-mock
PYTHONPATH=src .venv-open3d/bin/python scripts/view_3d.py replay SNAPSHOT_FOLDER --speed 2
PYTHONPATH=src .venv-open3d/bin/python scripts/view_3d.py recorded-pair \
  --left LEFT.png --right RIGHT.png --calibration calibration.json
PYTHONPATH=src .venv/bin/python scripts/view_3d.py disabled --frames 100
PYTHONPATH=src .venv/bin/python scripts/generate_3d_mock_sequence.py
PYTHONPATH=src .venv/bin/python scripts/benchmark_3d_viewer.py
```

Saved `.npz` snapshots retain point clouds, paths, poses, rigid transforms,
warnings, and revision metadata for deterministic replay. The headless sequence
command also creates top/side projections for environments without a display.
