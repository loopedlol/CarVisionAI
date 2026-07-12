"""End-to-end mock stereo, ToF, map, planning, and trajectory snapshots."""

from __future__ import annotations

import numpy as np

from .control import Pose2D
from .frames import CAMERA_OPTICAL_TO_VEHICLE, camera_to_vehicle
from .mapping import OCCUPIED
from .mock import make_noisy_multisensor_scene
from .planning import PlannerConfig, cell_to_metric, plan_candidates, prepare_planning_grid
from .planning_mock import make_planning_scenario
from .stereo import backproject_depth, disparity_to_depth, valid_points
from .tof import ranges_to_vehicle_points
from .trajectory import TrajectoryConfig, generate_trajectory
from .visualization3d import SCHEMA_VERSION, VisualizationSnapshot


def make_mock_visualization_sequence(frame_count: int = 40, *, seed: int = 12
                                     ) -> tuple[VisualizationSnapshot,...]:
    if frame_count<1:raise ValueError("frame_count must be positive")
    scene=make_noisy_multisensor_scene(180,240,seed=seed)
    depth=disparity_to_depth(scene.disparity_px,scene.calibration,min_depth_m=.3,max_depth_m=15)
    camera_dense=backproject_depth(depth,scene.calibration);camera_points=valid_points(camera_dense)
    stereo=valid_points(camera_to_vehicle(camera_dense));tof=ranges_to_vehicle_points(scene.tof_ranges_m,scene.tof_azimuth_rad,max_range_m=15)
    scenario=make_planning_scenario("blocked_direct");planner=PlannerConfig(robot_radius_m=.2,candidate_count=4)
    prepared=prepare_planning_grid(scenario.rough_data,scenario.config,planner=planner);candidates=plan_candidates(prepared,scenario.start,scenario.goal,planner=planner)
    trajectory=generate_trajectory(candidates[0],prepared,TrajectoryConfig(robot_radius_m=.2,safety_margin_m=.05,sample_spacing_m=.05))
    candidate_paths=tuple(np.asarray([(*cell_to_metric(cell,scenario.config),.03) for cell in candidate.cells]) for candidate in candidates)
    selected=candidate_paths[0];trajectory_points=np.asarray([(s.x_m,s.y_m,.05) for s in trajectory.samples])
    occupied=np.argwhere(scenario.rough_data==OCCUPIED);obstacles=np.asarray([(*cell_to_metric(tuple(cell),scenario.config),.12) for cell in occupied])
    angle=np.linspace(0,2*np.pi,33);footprint=np.column_stack((.2*np.cos(angle),.2*np.sin(angle),np.full(len(angle),.03)))
    snapshots=[];traveled=[]
    for index in range(frame_count):
        progress=index/max(frame_count-1,1);pose=Pose2D(.12*np.sin(progress*2*np.pi),progress*1.2,.04*np.sin(progress*np.pi))
        traveled.append((pose.x_m,pose.y_m,.06));visible_stereo=stereo[(np.arange(len(stereo))+index)%5!=0]
        moving_tof=tof.copy();moving_tof[:,2]+=.03*np.sin(index*.3)
        combined=np.vstack((visible_stereo,moving_tof));confidence=np.concatenate((np.full(len(visible_stereo),.8),np.full(len(moving_tof),.65)))
        changed=obstacles[(np.arange(len(obstacles))+index)%17==0]
        snapshots.append(VisualizationSnapshot(SCHEMA_VERSION,index/12,index,index//10,
            stereo_camera_points=camera_points,stereo_vehicle_points=visible_stereo,
            tof_sensor_points=moving_tof,tof_vehicle_points=moving_tof,
            combined_vehicle_points=combined,confidence=confidence,
            sensor_origins_vehicle=np.array([[0,.12,.32],[0,0,.25]]),
            camera_to_vehicle=CAMERA_OPTICAL_TO_VEHICLE,vehicle_pose=pose,
            estimated_pose=pose,ground_truth_pose=Pose2D(pose.x_m+.01*np.sin(index),pose.y_m,pose.heading_rad),
            robot_footprint_vehicle=footprint,path_candidates_vehicle=candidate_paths,
            selected_path_vehicle=selected,trajectory_vehicle=trajectory_points,
            traveled_trajectory_world=np.asarray(traveled),inspection_target_vehicle=np.array([[.6,3.,.4]]),
            occupancy_obstacles_vehicle=obstacles,changed_cells_vehicle=changed,
            warnings=("mock confidence reduction",) if index%13==0 else ()))
    return tuple(snapshots)
