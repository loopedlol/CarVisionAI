from dataclasses import replace
from pathlib import Path
from time import sleep

import numpy as np
import pytest

from carvision.control import Pose2D
from carvision.frames import RigidTransform
from carvision.visualization3d import (HeadlessRenderer, SCHEMA_VERSION, SnapshotQueue,
    SnapshotReplay, ViewerConfig, ViewerControls, ViewerService, VisualizationSnapshot,
    filter_and_downsample, load_snapshot, prepare_snapshot, save_snapshot, validate_snapshot)
from carvision.visualization_mock import make_mock_visualization_sequence


def snapshot(identifier=1, **kwargs):
    return VisualizationSnapshot(SCHEMA_VERSION, identifier*.1, identifier, 2, **kwargs)


def test_snapshot_validation_and_malformed_shapes():
    assert validate_snapshot(snapshot(stereo_vehicle_points=np.zeros((3,3))))==()
    errors=validate_snapshot(snapshot(stereo_vehicle_points=np.zeros((3,4))))
    assert "stereo_vehicle_points must be Nx2 or Nx3" in errors
    assert validate_snapshot(replace(snapshot(),schema_version=99))


def test_coordinate_conversion_matches_x_right_y_forward_z_up():
    item=snapshot(stereo_vehicle_points=np.array([[1.,0.,2.],[0.,1.,3.]]),
                  estimated_pose=Pose2D(10,20,np.pi/2),ground_truth_pose=Pose2D(10.1,20,np.pi/2))
    geometry=prepare_snapshot(item,ViewerConfig(voxel_size_m=.001),ViewerControls(fixed_world=True))
    np.testing.assert_allclose(geometry.points["stereo"],[[10,19,2],[11,20,3]],atol=1e-9)
    axes=geometry.lines["vehicle_axes"][0]
    assert axes[1,1] < 20  # local +x/right rotated with the pose
    assert axes[2,0] > 10  # local +y/forward points toward world +x
    assert axes[3,2] > 0   # +z remains up
    assert {"estimated_pose","ground_truth_pose"}<=set(geometry.lines)


def test_nonfinite_confidence_voxel_and_maximum_filtering():
    points=np.array([[0,0,0],[.01,.01,0],[1,1,1],[np.nan,0,0],[2,2,2]],float)
    confidence=np.array([1,.9,.1,1,.8])
    output,indices=filter_and_downsample(points,.1,2,confidence,.2)
    assert len(output)==2 and indices.tolist()==[0,4]
    assert np.isfinite(output).all()


def test_downsampling_is_deterministic_at_large_limit():
    rng=np.random.default_rng(3);points=rng.normal(size=(10000,3))
    first=filter_and_downsample(points,.001,1234)[0]
    second=filter_and_downsample(points,.001,1234)[0]
    np.testing.assert_array_equal(first,second)


def test_queue_drops_old_frames_and_prefers_newest():
    queue=SnapshotQueue(2)
    for index in range(5):queue.publish(snapshot(index))
    assert queue.newest().snapshot_id==4
    assert queue.dropped==4 and queue.consumed==1 and len(queue)==0


def test_queue_shutdown_rejects_publish():
    queue=SnapshotQueue();queue.close()
    assert queue.newest() is None and not queue.publish(snapshot())


def test_visibility_toggles_and_confidence_geometry():
    item=snapshot(stereo_vehicle_points=np.ones((2,3)),tof_vehicle_points=np.ones((3,3)),
                  combined_vehicle_points=np.arange(12,dtype=float).reshape(4,3),
                  confidence=np.array([0,.3,.9,1.]))
    controls=ViewerControls(show_stereo=False,show_tof=True,show_combined=True,
                            confidence_filter=True)
    geometry=prepare_snapshot(item,ViewerConfig(voxel_size_m=.001,confidence_threshold=.5),controls)
    assert "stereo" not in geometry.points and len(geometry.points["tof"])==1
    assert len(geometry.points["combined"])==2
    controls.toggle("show_stereo");assert controls.show_stereo
    with pytest.raises(ValueError):controls.toggle("requested_view")


def test_geometry_paths_footprint_grid_and_empty_snapshot():
    empty=prepare_snapshot(snapshot(),ViewerConfig())
    assert empty.points=={} and "ground_grid" in empty.lines
    item=snapshot(robot_footprint_vehicle=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]),
        path_candidates_vehicle=(np.array([[0,0],[0,1]]),),selected_path_vehicle=np.array([[0,0],[1,1]]),
        trajectory_vehicle=np.array([[0,0],[.5,.5]]),inspection_target_vehicle=np.array([[1,2,0]]))
    geometry=prepare_snapshot(item,ViewerConfig())
    assert {"footprint","candidate_0","selected_path","trajectory","inspection"}<=set(geometry.lines)


def test_snapshot_round_trip_and_replay_order_controls(tmp_path):
    paths=[]
    for identifier in (3,1,2):
        path=tmp_path/f"{identifier}.npz";save_snapshot(snapshot(identifier,stereo_vehicle_points=np.ones((identifier,3)),camera_to_vehicle=RigidTransform(translation_m=np.array([1.,2.,3.]))),path);paths.append(path)
    loaded=load_snapshot(paths[0]);assert loaded.snapshot_id==3
    np.testing.assert_array_equal(loaded.camera_to_vehicle.translation_m,[1,2,3])
    replay=SnapshotReplay(paths,2.0)
    assert replay.current().snapshot_id==1 and replay.next().snapshot_id==2
    assert replay.previous().snapshot_id==1 and replay.restart().snapshot_id==1
    replay.toggle_pause();assert replay.paused


def test_headless_service_isolated_and_reports_statistics():
    renderer=HeadlessRenderer();service=ViewerService(ViewerConfig(target_render_hz=200),renderer)
    service.start()
    for index in range(20):service.publish(snapshot(index,stereo_vehicle_points=np.ones((10,3))))
    for _ in range(50):
        if renderer.frames:break
        sleep(.005)
    service.stop();stats=service.statistics()
    assert renderer.frames and stats.published==20 and stats.dropped>0 and stats.closed


def test_disabled_viewer_has_zero_pipeline_cost_and_no_thread():
    service=ViewerService(ViewerConfig(enabled=False),HeadlessRenderer())
    assert not service.start() and not service.publish(snapshot())
    service.stop();assert service.statistics().published==0


def test_malformed_snapshot_does_not_crash_service():
    renderer=HeadlessRenderer();service=ViewerService(ViewerConfig(target_render_hz=200),renderer);service.start()
    service.publish(snapshot(stereo_vehicle_points=np.ones((2,4))))
    sleep(.03);service.stop()
    assert service.statistics().malformed>=1


def test_mock_replay_is_deterministic_and_changes_over_time():
    first=make_mock_visualization_sequence(4,seed=22);second=make_mock_visualization_sequence(4,seed=22)
    assert [item.snapshot_id for item in first]==[0,1,2,3]
    np.testing.assert_array_equal(first[2].combined_vehicle_points,second[2].combined_vehicle_points)
    assert first[0].vehicle_pose!=first[-1].vehicle_pose
