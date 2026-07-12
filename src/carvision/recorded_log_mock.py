"""Generated recorded-log fixtures for software tests and CLI demonstrations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .recorded_logs import RAW_COLUMNS


def generate_fixture_dataset(directory: Path, *, run_type: str = "straight", seed: int = 7,
                             duration_s: float = 12.0, dataset_id: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True); rng=np.random.default_rng(seed)
    dt=.02; radius=.065; track=.32; ticks=2048; modulus=65536; count_scale=ticks/(2*np.pi)
    rows=[]; left_total=right_total=65000.; x=y=heading=0.; distance=0.; heading_change=0.
    for index,t in enumerate(np.arange(0,duration_s+1e-9,dt)):
        if run_type=="stationary": v=omega=0.
        elif run_type=="in_place_rotation": v=0.;omega=.35 if .5<t<duration_s-1 else 0.
        elif run_type=="constant_radius_turn": v=.35 if .5<t<duration_s-1 else 0.;omega=.18 if v else 0.
        else:
            v=.45 if .5<t<duration_s-1 else 0.;omega=0.
        wl=(v+omega*track/2)/radius;wr=(v-omega*track/2)/radius
        left_total+=wl*dt*count_scale;right_total+=wr*dt*count_scale
        heading_mid=heading+omega*dt/2;x+=v*np.sin(heading_mid)*dt;y+=v*np.cos(heading_mid)*dt
        heading=(heading+omega*dt+np.pi)%(2*np.pi)-np.pi;distance+=abs(v)*dt;heading_change+=omega*dt
        encoder_valid=not (3.0<=t<3.2);imu_valid=not (5.0<=t<5.15)
        external=index%25==0
        row={name:"" for name in RAW_COLUMNS};row.update({
            "timestamp_s":f"{t:.6f}","clock_id":"monotonic","left_count":str(int(left_total)%modulus),
            "right_count":str(int(right_total)%modulus),"encoder_valid":str(int(encoder_valid)),
            "imu_yaw_rate_rps":f"{omega+.012+rng.normal(0,.008):.8f}","imu_valid":str(int(imu_valid)),
            "imu_heading_rad":f"{heading+rng.normal(0,.015):.8f}","heading_valid":str(int(index%10==0)),
            "external_valid":str(int(external)),"battery_v":f"{12.4-.01*t:.4f}",
            "command_v_mps":f"{v:.6f}","command_omega_rps":f"{omega:.6f}",
            "marker":"start" if index==0 else ("stop" if abs(t-(duration_s-1))<dt/2 else ""),
            "notes":"generated fixture; not real sensor data" if index==0 else ""})
        if external:
            row.update({"gt_x_m":f"{x+rng.normal(0,.025):.8f}",
                        "gt_y_m":f"{y+rng.normal(0,.025):.8f}",
                        "gt_heading_rad":f"{heading+rng.normal(0,.015):.8f}"})
        rows.append(row)
    with (directory/"raw.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=RAW_COLUMNS);writer.writeheader();writer.writerows(rows)
    manifest={"format_version":1,"dataset_id":dataset_id or f"fixture_{run_type}","raw_csv":"raw.csv",
              "processed_csv":"processed.csv","run_type":run_type,"units":{
                  "timestamp_s":"s","left_count":"tick","right_count":"tick",
                  "left_wheel_rad_s":"rad/s","right_wheel_rad_s":"rad/s",
                  "imu_yaw_rate_rps":"rad/s","imu_heading_rad":"rad","gt_x_m":"m","gt_y_m":"m",
                  "gt_heading_rad":"rad","battery_v":"V","command_v_mps":"m/s","command_omega_rps":"rad/s"},
              "clocks":{"monotonic":"host monotonic seconds; fixture shared clock"},
              "wheel_radius_m":radius,"track_width_m":track,"encoder_ticks_per_revolution":ticks,
              "encoder_counter_modulus":modulus,"left_direction_sign":1,"right_direction_sign":1,
              "imu_frame":"vehicle: +yaw toward +x from +y","external_frame":"local vehicle-start frame",
              "calibration_version":"fixture-v1","surface":"synthetic","payload_kg":2.0,
              "notes":"Generated fixture validates ingestion only.",
              "reference_distance_m":distance if run_type=="straight" else None,
              "reference_heading_change_rad":heading_change if "turn" in run_type or "rotation" in run_type else None}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return directory
