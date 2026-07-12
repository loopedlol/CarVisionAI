import csv
import json
from pathlib import Path

import numpy as np
import pytest

from carvision.pose_estimation import EstimatorMode
from carvision.recorded_log_mock import generate_fixture_dataset
from carvision.recorded_logs import (DatasetError, RAW_COLUMNS, analyze_calibration,
                                    load_manifest, preprocess_dataset, replay_dataset,
                                    tune_parameters, validate_dataset, write_report)


def fixture(tmp_path: Path, run_type: str = "straight", name: str = "data") -> Path:
    return generate_fixture_dataset(tmp_path / name, run_type=run_type, duration_s=4,
                                    dataset_id=name)


def test_manifest_csv_parse_units_and_missing_representation(tmp_path):
    path=fixture(tmp_path); manifest=load_manifest(path); report=validate_dataset(path)
    assert manifest.format_version==1 and report.valid and report.rows==201
    rows=list(csv.DictReader((path/"raw.csv").open()))
    assert rows[1]["left_wheel_rad_s"]==""


def test_inconsistent_units_rejected(tmp_path):
    path=fixture(tmp_path); data=json.loads((path/"manifest.json").read_text());data["units"]["imu_yaw_rate_rps"]="deg/s"
    (path/"manifest.json").write_text(json.dumps(data))
    with pytest.raises(DatasetError):load_manifest(path)


def test_missing_columns_malformed_csv(tmp_path):
    path=fixture(tmp_path); rows=list(csv.DictReader((path/"raw.csv").open()))
    with (path/"raw.csv").open("w",newline="") as handle:
        fields=list(RAW_COLUMNS);fields.remove("imu_valid");writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows([{k:v for k,v in row.items() if k in fields} for row in rows])
    report=validate_dataset(path);assert not report.valid and report.issues[0].code=="missing_columns"


def test_timestamp_order_duplicate_and_unknown_clock(tmp_path):
    path=fixture(tmp_path); rows=list(csv.DictReader((path/"raw.csv").open()));rows[3]["timestamp_s"]=rows[2]["timestamp_s"];rows[4]["clock_id"]="other"
    with (path/"raw.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=RAW_COLUMNS);writer.writeheader();writer.writerows(rows)
    codes={i.code for i in validate_dataset(path).issues};assert {"duplicate_timestamp","unknown_clock"}<=codes


def test_encoder_wrap_unwrap_count_conversion_and_dropout(tmp_path):
    path=fixture(tmp_path); report=preprocess_dataset(path)
    assert report.wrap_events>=2 and report.encoder_dropouts_s
    rows=list(csv.DictReader((path/"processed.csv").open()))
    speeds=[abs(float(r["left_wheel_rad_s"])) for r in rows if r["left_wheel_rad_s"]]
    assert max(speeds)<10 and np.median(speeds)>5


def test_stationary_gyro_bias_noise_and_allan(tmp_path):
    path=fixture(tmp_path,"stationary");preprocess_dataset(path);analysis=analyze_calibration(path)
    assert analysis.stationary_gyro_bias_rps==pytest.approx(.012,abs=.003)
    assert analysis.stationary_gyro_std_rps>0 and analysis.allan_deviation


def test_wheel_scale_calibration(tmp_path):
    path=fixture(tmp_path,"straight");preprocess_dataset(path);analysis=analyze_calibration(path)
    assert analysis.effective_left_radius_m==pytest.approx(.065,rel=.03)
    assert analysis.effective_right_radius_m==pytest.approx(.065,rel=.03)
    assert analysis.left_right_scale_ratio==pytest.approx(1,rel=.02)


def test_track_width_calibration(tmp_path):
    path=fixture(tmp_path,"in_place_rotation");preprocess_dataset(path);analysis=analyze_calibration(path)
    assert analysis.track_width_m==pytest.approx(.32,rel=.03)


def test_asynchronous_replay_modes_metrics_and_gating(tmp_path):
    path=fixture(tmp_path);preprocess_dataset(path)
    results=[replay_dataset(path,mode) for mode in EstimatorMode]
    assert all(result.metrics.samples==201 for result in results)
    assert results[-1].metrics.accepted_corrections>0
    assert results[-1].metrics.position_rmse_m is not None
    assert results[-1].timestamps_s==tuple(sorted(results[-1].timestamps_s))


def test_parameter_bounds_and_train_evaluation_separation(tmp_path):
    train=fixture(tmp_path,name="train");evaluation=fixture(tmp_path,name="evaluation",run_type="constant_radius_turn")
    preprocess_dataset(train);preprocess_dataset(evaluation)
    result=tune_parameters([train],[evaluation],max_candidates=5)
    assert result["candidate_count"]==5 and "evaluation" in result
    with pytest.raises(ValueError):tune_parameters([train],[train])


def test_deterministic_report(tmp_path):
    path=fixture(tmp_path);preprocess_dataset(path)
    first=write_report([path],tmp_path/"report1");second=write_report([path],tmp_path/"report2")
    # Runtime is expected to vary; all computed calibration/error values are deterministic.
    for report in (first,second):
        for value in report["datasets"]["data"]["replay"].values():value.pop("runtime_ms")
    assert first==second
    assert (tmp_path/"report1"/"data.png").exists()
