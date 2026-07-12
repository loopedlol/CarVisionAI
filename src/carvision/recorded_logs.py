"""Recorded timestamped sensor-log ingestion, calibration, and estimator replay.

Format version 1 uses ``manifest.json`` plus an asynchronous ``raw.csv``. Each
row may contain any subset of sensors; empty strings mean missing, while every
present sensor has an explicit ``*_valid`` flag. Raw files are never modified.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

from .control import Pose2D, WheelSpeeds, wrap_angle
from .pose_estimation import (EncoderMeasurement, EstimatorConfig, EstimatorHealth,
                              EstimatorMode, ExternalPoseMeasurement, HeadingMeasurement,
                              ImuYawRateMeasurement, PoseEstimator)

RAW_COLUMNS = ("timestamp_s", "clock_id", "left_count", "right_count", "encoder_valid",
               "left_wheel_rad_s", "right_wheel_rad_s", "imu_yaw_rate_rps", "imu_valid",
               "imu_heading_rad", "heading_valid", "gt_x_m", "gt_y_m", "gt_heading_rad",
               "external_valid", "battery_v", "command_v_mps", "command_omega_rps",
               "marker", "notes")
PROCESSED_COLUMNS = ("timestamp_s", "left_wheel_rad_s", "right_wheel_rad_s",
                     "encoder_valid", "imu_yaw_rate_rps", "imu_valid", "imu_heading_rad",
                     "heading_valid", "gt_x_m", "gt_y_m", "gt_heading_rad",
                     "external_valid", "battery_v", "command_v_mps", "command_omega_rps",
                     "marker", "notes")
EXPECTED_UNITS = {"timestamp_s": "s", "left_count": "tick", "right_count": "tick",
                  "left_wheel_rad_s": "rad/s", "right_wheel_rad_s": "rad/s",
                  "imu_yaw_rate_rps": "rad/s", "imu_heading_rad": "rad",
                  "gt_x_m": "m", "gt_y_m": "m", "gt_heading_rad": "rad",
                  "battery_v": "V", "command_v_mps": "m/s", "command_omega_rps": "rad/s"}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetManifest:
    format_version: int
    dataset_id: str
    raw_csv: str
    processed_csv: str
    run_type: str
    units: dict[str, str]
    clocks: dict[str, Any]
    wheel_radius_m: float
    track_width_m: float
    encoder_ticks_per_revolution: int
    encoder_counter_modulus: int | None
    left_direction_sign: int
    right_direction_sign: int
    imu_frame: str
    external_frame: str | None
    calibration_version: str
    surface: str
    payload_kg: float
    notes: str = ""
    reference_distance_m: float | None = None
    reference_heading_change_rad: float | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    row: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    rows: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class PreprocessReport:
    output_csv: str
    rows_read: int
    rows_written: int
    rejected_rows: int
    duplicate_rows: int
    wrap_events: int
    encoder_dropouts_s: tuple[tuple[float, float], ...]
    imu_dropouts_s: tuple[tuple[float, float], ...]
    estimated_imu_offset_s: float | None


@dataclass(frozen=True)
class CalibrationAnalysis:
    stationary_gyro_bias_rps: float | None
    stationary_gyro_std_rps: float | None
    allan_deviation: tuple[tuple[float, float], ...]
    left_distance_per_tick_m: float | None
    right_distance_per_tick_m: float | None
    effective_left_radius_m: float | None
    effective_right_radius_m: float | None
    track_width_m: float | None
    left_right_scale_ratio: float | None
    encoder_jitter_std_s: float | None
    imu_jitter_std_s: float | None
    command_latency_s: float | None
    stopping_response_s: float | None
    slip_ratio: float | None


@dataclass(frozen=True)
class ReplayMetrics:
    mode: str
    samples: int
    ground_truth_samples: int
    position_rmse_m: float | None
    final_position_error_m: float | None
    maximum_position_error_m: float | None
    heading_rmse_rad: float | None
    final_heading_error_rad: float | None
    maximum_heading_error_rad: float | None
    drift_per_m: float | None
    drift_per_minute_m: float | None
    accepted_corrections: int
    rejected_corrections: int
    total_encoder_dropout_s: float
    total_imu_dropout_s: float
    health_transitions: tuple[tuple[float, str], ...]
    final_position_std_m: float
    nees_mean: float | None
    nees_within_95_fraction: float | None
    runtime_ms: float


@dataclass(frozen=True)
class ReplayResult:
    metrics: ReplayMetrics
    timestamps_s: tuple[float, ...]
    poses: tuple[Pose2D, ...]
    ground_truth: tuple[Pose2D | None, ...]
    position_std_m: tuple[float, ...]
    innovations_accepted: tuple[tuple[float, bool], ...]


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    path = Path(dataset_dir) / "manifest.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"cannot read manifest: {error}") from error
    required = {field.name for field in DatasetManifest.__dataclass_fields__.values()
                if field.default is field.default_factory}
    # Dataclass construction provides a useful error for missing/extra fields.
    try:
        manifest = DatasetManifest(**data)
    except TypeError as error:
        raise DatasetError(f"invalid manifest fields: {error}") from error
    if manifest.format_version != 1:
        raise DatasetError("only dataset format_version 1 is supported")
    if manifest.run_type not in {"stationary", "straight", "constant_radius_turn",
                                 "in_place_rotation", "acceleration_braking", "general"}:
        raise DatasetError("unsupported run_type")
    if (manifest.wheel_radius_m <= 0 or manifest.track_width_m <= 0
            or manifest.encoder_ticks_per_revolution <= 0 or manifest.payload_kg < 0):
        raise DatasetError("vehicle dimensions, ticks/revolution, and payload are invalid")
    if manifest.left_direction_sign not in (-1, 1) or manifest.right_direction_sign not in (-1, 1):
        raise DatasetError("wheel direction signs must be -1 or +1")
    for name, unit in manifest.units.items():
        if name in EXPECTED_UNITS and EXPECTED_UNITS[name] != unit:
            raise DatasetError(f"unit for {name} must be {EXPECTED_UNITS[name]!r}, got {unit!r}")
    for name in ("timestamp_s", "imu_yaw_rate_rps"):
        if manifest.units.get(name) != EXPECTED_UNITS[name]:
            raise DatasetError(f"manifest units must declare {name} as {EXPECTED_UNITS[name]}")
    if len(manifest.clocks) > 1:
        for clock, definition in manifest.clocks.items():
            if not isinstance(definition, (int, float, dict)):
                raise DatasetError(f"multiple clock domains require an offset_s for {clock!r}")
            if isinstance(definition, dict) and "offset_s" not in definition:
                raise DatasetError(f"clock {clock!r} lacks offset_s")
    return manifest


def validate_dataset(dataset_dir: Path, *, jump_factor: float = 20.0,
                     ground_truth_jump_m: float = 2.0) -> ValidationReport:
    manifest = load_manifest(dataset_dir); path = Path(dataset_dir) / manifest.raw_csv
    issues: list[ValidationIssue] = []
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(RAW_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                return ValidationReport(False, 0, (ValidationIssue("error", "missing_columns",
                                                                  f"missing {sorted(missing)}"),))
            rows = list(reader)
    except OSError as error:
        raise DatasetError(f"cannot read raw CSV: {error}") from error
    last_by_clock: dict[str, float] = {}; spacings: dict[str, list[float]] = {}
    last_gt: Pose2D | None = None
    for number, row in enumerate(rows, 2):
        try: timestamp = float(row["timestamp_s"])
        except ValueError:
            issues.append(ValidationIssue("error", "invalid_timestamp", "non-numeric timestamp", number)); continue
        if not np.isfinite(timestamp):
            issues.append(ValidationIssue("error", "invalid_timestamp", "non-finite timestamp", number)); continue
        clock = row["clock_id"]
        if clock not in manifest.clocks:
            issues.append(ValidationIssue("error", "unknown_clock", f"clock {clock!r} undeclared", number))
        previous = last_by_clock.get(clock)
        if previous is not None:
            if timestamp < previous: issues.append(ValidationIssue("error", "timestamp_order", "timestamp decreased", number))
            elif timestamp == previous: issues.append(ValidationIssue("error", "duplicate_timestamp", "duplicate clock timestamp", number))
            else: spacings.setdefault(clock, []).append(timestamp - previous)
        last_by_clock[clock] = timestamp
        for field in RAW_COLUMNS[2:18]:
            if row[field] and field not in {"marker", "notes"}:
                try: value = float(row[field])
                except ValueError: issues.append(ValidationIssue("error", "non_numeric", f"{field} is non-numeric", number)); continue
                if not np.isfinite(value): issues.append(ValidationIssue("error", "nonfinite", f"{field} is non-finite", number))
        for flag in ("encoder_valid", "imu_valid", "heading_valid", "external_valid"):
            if row[flag] not in ("0", "1", ""):
                issues.append(ValidationIssue("error", "invalid_flag", f"{flag} must be 0, 1, or empty", number))
        if row["external_valid"] == "1" and all(row[key] for key in ("gt_x_m", "gt_y_m", "gt_heading_rad")):
            pose = Pose2D(float(row["gt_x_m"]), float(row["gt_y_m"]), float(row["gt_heading_rad"]))
            if last_gt and np.hypot(pose.x_m - last_gt.x_m, pose.y_m - last_gt.y_m) > ground_truth_jump_m:
                issues.append(ValidationIssue("warning", "ground_truth_jump", "large ground-truth discontinuity", number))
            last_gt = pose
    for clock, delta in spacings.items():
        if len(delta) > 3:
            median = np.median(delta)
            if max(delta) > jump_factor * median:
                issues.append(ValidationIssue("warning", "timestamp_gap", f"large gap on {clock}"))
    if len(set(last_by_clock)) > 1 and not manifest.clocks:
        issues.append(ValidationIssue("error", "clock_mismatch", "multiple clocks lack mapping"))
    return ValidationReport(not any(issue.severity == "error" for issue in issues), len(rows), tuple(issues))


def preprocess_dataset(dataset_dir: Path, *, dropout_factor: float = 5.0,
                       max_wheel_speed_rad_s: float = 40.0) -> PreprocessReport:
    dataset_dir = Path(dataset_dir); manifest = load_manifest(dataset_dir)
    validation = validate_dataset(dataset_dir)
    if not validation.valid: raise DatasetError("dataset validation failed; preprocessing refused")
    with (dataset_dir / manifest.raw_csv).open(newline="") as handle: rows = list(csv.DictReader(handle))
    events: list[dict[str, Any]] = []; last_counts = None; last_count_time = None
    count_offsets = [0, 0]; wraps = rejected = duplicates = 0; seen: set[tuple[str, float]] = set()
    encoder_times, imu_times = [], []
    modulus = manifest.encoder_counter_modulus
    for row in rows:
        raw_timestamp = float(row["timestamp_s"]); timestamp = raw_timestamp + _clock_offset(manifest, row["clock_id"]); key = (row["clock_id"], raw_timestamp)
        if key in seen: duplicates += 1; continue
        seen.add(key); output = {name: row.get(name, "") for name in PROCESSED_COLUMNS}
        output["timestamp_s"] = f"{timestamp:.9f}"
        if row["encoder_valid"] == "1":
            wheel_values: tuple[float, float] | None = None
            if row["left_wheel_rad_s"] and row["right_wheel_rad_s"]:
                wheel_values = (float(row["left_wheel_rad_s"]), float(row["right_wheel_rad_s"]))
            elif row["left_count"] and row["right_count"]:
                counts = [int(float(row["left_count"])), int(float(row["right_count"]))]
                if last_counts is not None and last_count_time is not None:
                    delta = [counts[i] - last_counts[i] for i in range(2)]
                    if modulus:
                        for i in range(2):
                            if delta[i] > modulus / 2: delta[i] -= modulus; wraps += 1
                            elif delta[i] < -modulus / 2: delta[i] += modulus; wraps += 1
                    dt = timestamp - last_count_time
                    scale = 2 * np.pi / manifest.encoder_ticks_per_revolution / dt
                    wheel_values = (delta[0] * scale * manifest.left_direction_sign,
                                    delta[1] * scale * manifest.right_direction_sign)
                last_counts, last_count_time = counts, timestamp
            if wheel_values is not None and max(abs(wheel_values[0]), abs(wheel_values[1])) <= max_wheel_speed_rad_s:
                output["left_wheel_rad_s"], output["right_wheel_rad_s"] = map(lambda v: f"{v:.9f}", wheel_values)
                encoder_times.append(timestamp)
            else:
                output["encoder_valid"] = "0"; rejected += int(wheel_values is not None)
        if row["imu_valid"] == "1": imu_times.append(timestamp)
        events.append(output)
    events.sort(key=lambda item: float(item["timestamp_s"]))
    output_path = dataset_dir / manifest.processed_csv
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_COLUMNS); writer.writeheader(); writer.writerows(events)
    encoder_dropouts = _dropouts(encoder_times, dropout_factor); imu_dropouts = _dropouts(imu_times, dropout_factor)
    offset = estimate_stream_offset(rows)
    return PreprocessReport(str(output_path), len(rows), len(events), rejected, duplicates, wraps,
                            tuple(encoder_dropouts), tuple(imu_dropouts), offset)


def analyze_calibration(dataset_dir: Path) -> CalibrationAnalysis:
    dataset_dir = Path(dataset_dir); manifest = load_manifest(dataset_dir)
    rows = _processed_rows(dataset_dir, manifest)
    stationary_rows = [r for r in rows if r["imu_valid"] == "1" and r["imu_yaw_rate_rps"]
                       and r["encoder_valid"] == "1" and r["left_wheel_rad_s"]
                       and abs(float(r["left_wheel_rad_s"]))
                       + abs(float(r["right_wheel_rad_s"])) < .2]
    imu = np.asarray([float(r["imu_yaw_rate_rps"]) for r in stationary_rows], float)
    imu_times = [float(r["timestamp_s"]) for r in stationary_rows]
    enc_rows = [r for r in rows if r["encoder_valid"] == "1" and r["left_wheel_rad_s"]]
    enc_times = [float(r["timestamp_s"]) for r in enc_rows]
    left_distance = right_distance = 0.0
    for first, second in zip(enc_rows, enc_rows[1:]):
        dt = float(second["timestamp_s"]) - float(first["timestamp_s"])
        left_distance += float(second["left_wheel_rad_s"]) * manifest.wheel_radius_m * dt
        right_distance += float(second["right_wheel_rad_s"]) * manifest.wheel_radius_m * dt
    left_tick = right_tick = left_radius = right_radius = track = ratio = None
    if manifest.reference_distance_m and enc_rows:
        # Scale current integrated distances to the independently measured run distance.
        left_scale = manifest.reference_distance_m / max(abs(left_distance), 1e-12)
        right_scale = manifest.reference_distance_m / max(abs(right_distance), 1e-12)
        left_radius = manifest.wheel_radius_m * left_scale; right_radius = manifest.wheel_radius_m * right_scale
        left_tick = 2 * np.pi * left_radius / manifest.encoder_ticks_per_revolution
        right_tick = 2 * np.pi * right_radius / manifest.encoder_ticks_per_revolution
        ratio = left_scale / right_scale
    if manifest.reference_heading_change_rad and abs(manifest.reference_heading_change_rad) > 1e-6:
        track = (left_distance - right_distance) / manifest.reference_heading_change_rad
        track = abs(track)
    command_latency = _command_latency(rows)
    stopping = _stopping_response(rows)
    slip = None
    if manifest.reference_distance_m is not None:
        gt_distance = manifest.reference_distance_m
        wheel_distance = (abs(left_distance) + abs(right_distance)) / 2
        if wheel_distance > 1e-9: slip = 1 - gt_distance / wheel_distance
    return CalibrationAnalysis(float(np.mean(imu)) if len(imu) else None,
                               float(np.std(imu, ddof=1)) if len(imu) > 1 else None,
                               tuple(_allan(imu, imu_times)), left_tick, right_tick, left_radius,
                               right_radius, track, ratio, _jitter(enc_times), _jitter(imu_times),
                               command_latency, stopping, slip)


def replay_dataset(dataset_dir: Path, mode: EstimatorMode,
                   estimator_config: EstimatorConfig | None = None) -> ReplayResult:
    dataset_dir = Path(dataset_dir); manifest = load_manifest(dataset_dir)
    rows = _processed_rows(dataset_dir, manifest)
    config = estimator_config or EstimatorConfig(wheel_radius_m=manifest.wheel_radius_m,
                                                 track_width_m=manifest.track_width_m)
    estimator = PoseEstimator(mode, config); first_time = float(rows[0]["timestamp_s"])
    interpolated_truth = _interpolated_truth(rows)
    initial_gt = interpolated_truth[0] or Pose2D(0, 0, 0)
    estimator.reset(initial_gt, first_time)
    times, poses, truths, stds, corrections, health = [], [], [], [], [], []
    begin = perf_counter()
    for index, row in enumerate(rows):
        timestamp = float(row["timestamp_s"])
        if row["encoder_valid"] == "1" and row["left_wheel_rad_s"]:
            estimator.process_encoder(EncoderMeasurement(timestamp, WheelSpeeds(
                float(row["left_wheel_rad_s"]), float(row["right_wheel_rad_s"]))))
        if row["imu_valid"] == "1" and row["imu_yaw_rate_rps"]:
            estimator.process_imu(ImuYawRateMeasurement(timestamp, float(row["imu_yaw_rate_rps"])))
        if row["heading_valid"] == "1" and row["imu_heading_rad"]:
            estimator.process_heading(HeadingMeasurement(timestamp, float(row["imu_heading_rad"])))
        if row["external_valid"] == "1":
            if mode == EstimatorMode.ENCODER_IMU_EXTERNAL:
                corrections.append((timestamp, estimator.process_external_pose(
                    ExternalPoseMeasurement(timestamp, _pose(row)))))
        estimator.advance_time(timestamp); status = estimator.status()
        times.append(timestamp); poses.append(status.pose); truths.append(interpolated_truth[index])
        stds.append(float(np.sqrt(max(status.covariance[0, 0], status.covariance[1, 1]))))
        if not health or health[-1][1] != status.health.value: health.append((timestamp, status.health.value))
    runtime = (perf_counter() - begin) * 1000
    preprocess = _dropout_metrics(rows)
    metrics = _replay_metrics(mode, times, poses, truths, stds, estimator, health, preprocess, runtime)
    return ReplayResult(metrics, tuple(times), tuple(poses), tuple(truths), tuple(stds), tuple(corrections))


def tune_parameters(training_dirs: list[Path], evaluation_dirs: list[Path], *,
                    max_candidates: int = 24) -> dict[str, Any]:
    if not training_dirs or not evaluation_dirs: raise ValueError("training and evaluation runs are both required")
    overlap = {Path(p).resolve() for p in training_dirs} & {Path(p).resolve() for p in evaluation_dirs}
    if overlap: raise ValueError("training and evaluation datasets must be disjoint")
    candidates = []
    for encoder_noise in (.04, .10, .20):
        for gyro_noise in (.015, .03):
            for external_noise in (.04, .10):
                for pose_gate in (7.815, 11.34):
                    candidates.append(EstimatorConfig(encoder_linear_noise_mps=encoder_noise,
                                                      gyro_noise_rps=gyro_noise,
                                                      external_position_noise_m=external_noise,
                                                      gate_chi2_pose=pose_gate))
    candidates = candidates[:max_candidates]
    scored = []
    for config in candidates:
        metrics = [replay_dataset(path, EstimatorMode.ENCODER_IMU_EXTERNAL, config).metrics
                   for path in training_dirs]
        score = np.mean([m.position_rmse_m if m.position_rmse_m is not None else 1e6 for m in metrics])
        scored.append((float(score), config))
    scored.sort(key=lambda item: item[0]); best_score, best = scored[0]
    evaluation = {Path(path).name: asdict(replay_dataset(path, EstimatorMode.ENCODER_IMU_EXTERNAL,
                                                         best).metrics)
                  for path in evaluation_dirs}
    return {"strategy": "bounded 3x2x2x2 covariance/gate grid on training RMSE; one untouched evaluation pass",
            "candidate_count": len(candidates), "training_score": best_score,
            "best_config": asdict(best), "evaluation": evaluation}


def write_report(dataset_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True); report: dict[str, Any] = {"generated_fixture_warning":
        "Synthetic fixtures validate software behavior only, not real sensor accuracy.", "datasets": {}}
    for directory in dataset_dirs:
        manifest = load_manifest(directory); calibration = analyze_calibration(directory)
        replays = {mode.value: replay_dataset(directory, mode) for mode in EstimatorMode}
        report["datasets"][manifest.dataset_id] = {"calibration": asdict(calibration),
            "replay": {name: asdict(result.metrics) for name, result in replays.items()}}
        render_replay_diagnostics(directory, replays, output_dir / f"{manifest.dataset_id}.png")
        render_sensor_diagnostics(directory, output_dir / f"{manifest.dataset_id}_sensors.png")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Recorded sensor evaluation", "",
             "> Generated fixtures validate tooling only; they do not validate real sensors.", ""]
    for name, data in report["datasets"].items():
        lines += [f"## {name}", "", "| Mode | Position RMSE | Final error | Heading RMSE |", "|---|---:|---:|---:|"]
        for mode, metric in data["replay"].items():
            lines.append(f"| {mode} | {_fmt(metric['position_rmse_m'])} | {_fmt(metric['final_position_error_m'])} | {_fmt(metric['heading_rmse_rad'])} |")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return report


def render_replay_diagnostics(dataset_dir: Path, results: dict[str, ReplayResult], output: Path) -> None:
    image = np.full((900, 1400, 3), 248, np.uint8); colors = [(30, 80, 210), (20, 150, 50), (180, 60, 160)]
    cv2.putText(image, Path(dataset_dir).name, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (20, 20, 20), 2)
    all_points = []; gt_points = []
    for result in results.values(): all_points.append(np.asarray([(p.x_m, p.y_m) for p in result.poses]))
    external = next(iter(results.values())).ground_truth
    gt_points = np.asarray([(p.x_m, p.y_m) for p in external if p is not None])
    values = np.vstack(all_points + ([gt_points] if len(gt_points) else [])); low, high = values.min(0), values.max(0)
    span = np.maximum(high-low, .1); low -= .1*span; high += .1*span
    def pixels(points): return np.column_stack((40+(points[:,0]-low[0])/(high[0]-low[0])*620,
                                                840-(points[:,1]-low[1])/(high[1]-low[1])*760)).astype(np.int32)
    if len(gt_points): cv2.polylines(image, [pixels(gt_points)], False, (20,20,20), 4)
    for result, color in zip(results.values(), colors):
        cv2.polylines(image, [pixels(np.asarray([(p.x_m,p.y_m) for p in result.poses]))], False, color, 2)
    y = 80
    for (name, result), color in zip(results.items(), colors):
        errors = np.asarray([np.hypot(p.x_m-g.x_m,p.y_m-g.y_m) if g else np.nan
                             for p,g in zip(result.poses,result.ground_truth)])
        _line_chart(image, (730,y,630,180), np.asarray(result.timestamps_s), errors, color, f"{name} position error m")
        y += 220
    cv2.imwrite(str(output), image)


def render_sensor_diagnostics(dataset_dir: Path, output: Path) -> None:
    manifest=load_manifest(dataset_dir);rows=_processed_rows(Path(dataset_dir),manifest)
    image=np.full((900,1400,3),248,np.uint8);cv2.putText(image,f"{manifest.dataset_id}: processed sensors",(25,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
    enc=[r for r in rows if r["encoder_valid"]=="1" and r["left_wheel_rad_s"]];imu=[r for r in rows if r["imu_valid"]=="1" and r["imu_yaw_rate_rps"]]
    et=np.asarray([float(r["timestamp_s"]) for r in enc]);left=np.asarray([float(r["left_wheel_rad_s"]) for r in enc]);right=np.asarray([float(r["right_wheel_rad_s"]) for r in enc])
    it=np.asarray([float(r["timestamp_s"]) for r in imu]);yaw=np.asarray([float(r["imu_yaw_rate_rps"]) for r in imu]);bias=analyze_calibration(dataset_dir).stationary_gyro_bias_rps
    _two_line_chart(image,(40,70,1320,180),et,left,right,(30,80,210),(20,150,50),"processed wheel speeds rad/s: left red, right green")
    _line_chart(image,(40,280,1320,180),it,yaw,(180,60,160),f"IMU yaw rate rad/s; stationary bias={_fmt(bias)}")
    if len(et)>1:_line_chart(image,(40,490,1320,150),et[1:],np.diff(et),(30,80,210),"encoder timestamp spacing s (dropouts appear as spikes)")
    if len(it)>1:_line_chart(image,(40,670,1320,150),it[1:],np.diff(it),(180,60,160),"IMU timestamp spacing s (dropouts appear as spikes)")
    cv2.imwrite(str(output),image)


def estimate_stream_offset(rows: list[dict[str, str]], max_offset_s: float = .5) -> float | None:
    command = [(float(r["timestamp_s"]), float(r["command_omega_rps"])) for r in rows if r.get("command_omega_rps")]
    imu = [(float(r["timestamp_s"]), float(r["imu_yaw_rate_rps"])) for r in rows if r.get("imu_yaw_rate_rps")]
    if len(command) < 10 or len(imu) < 10: return None
    start = max(command[0][0], imu[0][0]); end = min(command[-1][0], imu[-1][0])
    if end <= start: return None
    grid = np.linspace(start, end, 500); command_values = np.interp(grid, *zip(*command)); imu_values = np.interp(grid, *zip(*imu))
    if np.std(command_values) < 1e-9 or np.std(imu_values) < 1e-9:
        return None
    offsets = np.linspace(-max_offset_s, max_offset_s, 101); scores = []
    for offset in offsets:
        shifted = np.interp(grid + offset, grid, imu_values, left=np.nan, right=np.nan); mask=np.isfinite(shifted)
        scores.append(np.corrcoef(command_values[mask], shifted[mask])[0,1] if mask.sum()>10 else -np.inf)
    scores_array = np.asarray(scores)
    if not np.isfinite(scores_array).any(): return None
    return float(offsets[int(np.nanargmax(scores_array))])


def _processed_rows(directory: Path, manifest: DatasetManifest) -> list[dict[str, str]]:
    path = directory / manifest.processed_csv
    if not path.exists(): preprocess_dataset(directory)
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def _clock_offset(manifest: DatasetManifest, clock: str) -> float:
    definition=manifest.clocks[clock]
    if isinstance(definition,(int,float)):return float(definition)
    if isinstance(definition,dict):return float(definition.get("offset_s",0.0))
    return 0.0


def _pose(row: dict[str, str]) -> Pose2D:
    return Pose2D(float(row["gt_x_m"]), float(row["gt_y_m"]), wrap_angle(float(row["gt_heading_rad"])))


def _dropouts(times: list[float], factor: float) -> list[tuple[float,float]]:
    if len(times)<3: return []
    median=float(np.median(np.diff(times))); return [(a,b) for a,b in zip(times,times[1:]) if b-a>factor*median]


def _dropout_metrics(rows):
    enc=[float(r["timestamp_s"]) for r in rows if r["encoder_valid"]=="1"]
    imu=[float(r["timestamp_s"]) for r in rows if r["imu_valid"]=="1"]
    return sum(b-a for a,b in _dropouts(enc,5)), sum(b-a for a,b in _dropouts(imu,5))


def _jitter(times):
    if len(times)<=2:return None
    delta=np.diff(times);median=np.median(delta);regular=delta[delta<=5*median]
    return float(np.std(regular-median)) if len(regular) else None


def _interpolated_truth(rows: list[dict[str, str]]) -> list[Pose2D | None]:
    references=[(float(r["timestamp_s"]),_pose(r)) for r in rows if r["external_valid"]=="1"]
    if len(references)<2:return [None]*len(rows)
    reference_time=np.asarray([item[0] for item in references]);x=np.asarray([item[1].x_m for item in references]);y=np.asarray([item[1].y_m for item in references]);heading=np.unwrap([item[1].heading_rad for item in references])
    times=np.asarray([float(r["timestamp_s"]) for r in rows]);mask=(times>=reference_time[0])&(times<=reference_time[-1])
    output:list[Pose2D|None]=[None]*len(rows)
    for index in np.where(mask)[0]:output[index]=Pose2D(float(np.interp(times[index],reference_time,x)),float(np.interp(times[index],reference_time,y)),wrap_angle(float(np.interp(times[index],reference_time,heading))))
    return output


def _allan(values, times):
    if len(values)<8 or len(times)<8: return []
    dt=float(np.median(np.diff(times))); output=[]
    for size in (1,2,4,8,16,32):
        if 2*size>=len(values): break
        means=np.asarray([np.mean(values[i:i+size]) for i in range(0,len(values)-size+1,size)])
        output.append((size*dt,float(np.sqrt(.5*np.mean(np.diff(means)**2)))))
    return output


def _command_latency(rows):
    samples=[(float(r["timestamp_s"]),float(r["command_v_mps"]),
              (float(r["left_wheel_rad_s"])+float(r["right_wheel_rad_s"]))/2)
             for r in rows if r["command_v_mps"] and r["left_wheel_rad_s"]]
    if len(samples)<4:return None
    times=np.asarray([s[0] for s in samples]); command=np.asarray([s[1] for s in samples]); motion=np.asarray([s[2] for s in samples])
    c=np.where(np.abs(command)>max(.05,.1*np.max(np.abs(command))))[0]; m=np.where(np.abs(motion)>max(.2,.1*np.max(np.abs(motion))))[0]
    return float(times[m[0]]-times[c[0]]) if len(c) and len(m) else None


def _stopping_response(rows):
    samples=[(float(r["timestamp_s"]),float(r["command_v_mps"]),abs(float(r["left_wheel_rad_s"])+float(r["right_wheel_rad_s"])))
             for r in rows if r["command_v_mps"] and r["left_wheel_rad_s"]]
    for i in range(1,len(samples)):
        if abs(samples[i][1])<.01 and abs(samples[i-1][1])>=.01:
            for later in samples[i:]:
                if later[2]<.2:return later[0]-samples[i][0]
    return None


def _replay_metrics(mode,times,poses,truths,stds,estimator,health,dropout,runtime):
    valid=[(i,p,g) for i,(p,g) in enumerate(zip(poses,truths)) if g is not None]
    if valid:
        pe=np.asarray([np.hypot(p.x_m-g.x_m,p.y_m-g.y_m) for _,p,g in valid]); he=np.asarray([abs(wrap_angle(p.heading_rad-g.heading_rad)) for _,p,g in valid])
        distance=sum(np.hypot(b.x_m-a.x_m,b.y_m-a.y_m) for a,b in zip(poses,poses[1:])); duration=max(times[-1]-times[0],1e-9)
        nees=[]
        # Position-only normalized error using reported maximum-axis std.
        for (i,p,g),error in zip(valid,pe): nees.append(error**2/max(stds[i]**2,1e-12))
        drift_per_m=float(pe[-1]/distance) if distance>=.1 else None
        values=(float(np.sqrt(np.mean(pe**2))),float(pe[-1]),float(np.max(pe)),float(np.sqrt(np.mean(he**2))),float(he[-1]),float(np.max(he)),drift_per_m,float(pe[-1]/duration*60),float(np.mean(nees)),float(np.mean(np.asarray(nees)<=5.991)))
    else: values=(None,)*10
    s=estimator.status()
    return ReplayMetrics(mode.value,len(times),len(valid),*values[:8],s.accepted_external,s.rejected_outliers,dropout[0],dropout[1],tuple(health),stds[-1],values[8],values[9],runtime)


def _line_chart(image,box,time,values,color,label):
    x,y,w,h=box;cv2.rectangle(image,(x,y),(x+w,y+h),(210,210,210),1);cv2.putText(image,label,(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,30,30),1)
    mask=np.isfinite(values)
    if mask.sum()<2:return
    px=x+(time[mask]-time[0])/max(time[-1]-time[0],1e-9)*w;low=min(float(np.nanmin(values)),0.);high=max(float(np.nanmax(values)),0.);high=high if high-low>1e-9 else low+1;py=y+h-(values[mask]-low)/(high-low)*(h-25)
    cv2.polylines(image,[np.column_stack((px,py)).astype(np.int32)],False,color,2)


def _two_line_chart(image,box,time,first,second,first_color,second_color,label):
    x,y,w,h=box;cv2.rectangle(image,(x,y),(x+w,y+h),(210,210,210),1);cv2.putText(image,label,(x+8,y+18),cv2.FONT_HERSHEY_SIMPLEX,.42,(30,30,30),1)
    if len(time)<2:return
    low=min(float(first.min()),float(second.min()));high=max(float(first.max()),float(second.max()))
    if high-low<1e-9:high=low+1
    for values,color in ((first,first_color),(second,second_color)):
        px=x+(time-time[0])/max(time[-1]-time[0],1e-9)*w;py=y+h-(values-low)/(high-low)*(h-25);cv2.polylines(image,[np.column_stack((px,py)).astype(np.int32)],False,color,2)


def _fmt(value): return "n/a" if value is None else f"{value:.4f}"
