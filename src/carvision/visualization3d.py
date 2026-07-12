"""Optional, non-blocking 3D visualization boundary with a lazy Open3D backend.

Open3D display coordinates are used without remapping: red +X is vehicle-right,
green +Y is vehicle-forward, and blue +Z is up. This is a right-handed frame
(``x cross y = z``) and matches the rest of CarVision exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from time import monotonic, perf_counter, sleep
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import cv2

from .control import Pose2D
from .frames import RigidTransform

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VisualizationSnapshot:
    schema_version: int
    timestamp_s: float
    snapshot_id: int
    map_revision: int
    stereo_camera_points: NDArray[np.float64] | None = None
    stereo_vehicle_points: NDArray[np.float64] | None = None
    tof_sensor_points: NDArray[np.float64] | None = None
    tof_vehicle_points: NDArray[np.float64] | None = None
    combined_vehicle_points: NDArray[np.float64] | None = None
    stereo_colors_rgb: NDArray[np.float64] | None = None
    confidence: NDArray[np.float64] | None = None
    sensor_origins_vehicle: NDArray[np.float64] | None = None
    camera_to_vehicle: RigidTransform | None = None
    tof_to_vehicle: RigidTransform | None = None
    vehicle_pose: Pose2D | None = None
    estimated_pose: Pose2D | None = None
    ground_truth_pose: Pose2D | None = None
    robot_footprint_vehicle: NDArray[np.float64] | None = None
    path_candidates_vehicle: tuple[NDArray[np.float64], ...] = ()
    selected_path_vehicle: NDArray[np.float64] | None = None
    trajectory_vehicle: NDArray[np.float64] | None = None
    traveled_trajectory_world: NDArray[np.float64] | None = None
    inspection_target_vehicle: NDArray[np.float64] | None = None
    occupancy_obstacles_vehicle: NDArray[np.float64] | None = None
    changed_cells_vehicle: NDArray[np.float64] | None = None
    warnings: tuple[str, ...] = ()
    fault: str | None = None


@dataclass(frozen=True)
class ViewerConfig:
    enabled: bool = True
    queue_size: int = 2
    voxel_size_m: float = .035
    max_points: int = 100_000
    confidence_threshold: float = .25
    point_size: float = 2.0
    ground_grid_extent_m: float = 10.0
    ground_grid_spacing_m: float = 1.0
    target_render_hz: float = 30.0
    fixed_world: bool = True

    def __post_init__(self) -> None:
        if self.queue_size < 1 or self.max_points < 1 or min(self.voxel_size_m,
            self.point_size, self.ground_grid_extent_m, self.ground_grid_spacing_m,
            self.target_render_hz) <= 0:
            raise ValueError("viewer sizes, limits, and rates must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must lie in [0,1]")


@dataclass
class ViewerControls:
    paused: bool = False
    show_stereo: bool = True
    show_tof: bool = True
    show_combined: bool = False
    show_frames: bool = True
    show_paths: bool = True
    show_occupancy: bool = True
    confidence_filter: bool = True
    fixed_world: bool = True
    point_size: float = 2.0
    requested_view: str = "perspective"
    save_cloud_requested: bool = False
    screenshot_requested: bool = False
    replay_next_requested: bool = False
    replay_previous_requested: bool = False
    replay_restart_requested: bool = False

    def toggle(self, name: str) -> None:
        if not hasattr(self, name) or not isinstance(getattr(self, name), bool):
            raise ValueError(f"unknown boolean viewer control {name!r}")
        setattr(self, name, not getattr(self, name))


@dataclass(frozen=True)
class ViewerStatistics:
    published: int
    consumed: int
    dropped: int
    rendered: int
    malformed: int
    render_rate_hz: float
    last_prepare_ms: float
    queue_depth: int
    closed: bool


@dataclass(frozen=True)
class PreparedGeometry:
    snapshot_id: int
    timestamp_s: float
    points: dict[str, NDArray[np.float64]]
    colors: dict[str, NDArray[np.float64]]
    lines: dict[str, tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]]
    warnings: tuple[str, ...]
    fault: str | None
    total_input_points: int
    total_displayed_points: int


class SnapshotQueue:
    """Bounded handoff that never blocks publishers and consumes newest first."""

    def __init__(self, capacity: int = 2) -> None:
        if capacity < 1: raise ValueError("capacity must be positive")
        self.capacity=capacity;self._items:deque[VisualizationSnapshot]=deque();self._condition=Condition()
        self.published=self.consumed=self.dropped=0;self.closed=False

    def publish(self, snapshot: VisualizationSnapshot) -> bool:
        with self._condition:
            if self.closed:return False
            self.published+=1
            if len(self._items)>=self.capacity:self._items.popleft();self.dropped+=1
            self._items.append(snapshot);self._condition.notify();return True

    def newest(self, timeout_s: float = 0.0) -> VisualizationSnapshot | None:
        with self._condition:
            if not self._items and not self.closed and timeout_s>0:self._condition.wait(timeout_s)
            if not self._items:return None
            newest=self._items.pop();obsolete=len(self._items);self._items.clear();self.dropped+=obsolete;self.consumed+=1
            return newest

    def close(self) -> None:
        with self._condition:self.closed=True;self._items.clear();self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:return len(self._items)


class Renderer(Protocol):
    def open(self, controls: ViewerControls) -> bool: ...
    def update(self, geometry: PreparedGeometry, controls: ViewerControls) -> bool: ...
    def close(self) -> None: ...


class HeadlessRenderer:
    """Test renderer that records prepared frames and never opens a window."""
    def __init__(self) -> None:self.frames:list[PreparedGeometry]=[];self.is_open=False
    def open(self, controls: ViewerControls) -> bool:self.is_open=True;return True
    def update(self, geometry: PreparedGeometry, controls: ViewerControls) -> bool:
        if not self.is_open:return False
        self.frames.append(geometry);return True
    def close(self) -> None:self.is_open=False


class ViewerService:
    """Optional background consumer; producer work is only a deque insertion."""
    def __init__(self, config: ViewerConfig = ViewerConfig(), renderer: Renderer | None = None) -> None:
        self.config=config;self.controls=ViewerControls(fixed_world=config.fixed_world,point_size=config.point_size)
        self.queue=SnapshotQueue(config.queue_size);self.renderer=renderer or Open3DRenderer()
        self._stop=Event();self._thread:Thread|None=None;self._rendered=self._malformed=0
        self._begin=monotonic();self._last_prepare_ms=0.;self._lock=Lock()

    def start(self) -> bool:
        if not self.config.enabled:return False
        if self._thread and self._thread.is_alive():return True
        self._thread=Thread(target=self._run,name="carvision-3d-viewer",daemon=True);self._thread.start();return True

    def publish(self, snapshot: VisualizationSnapshot) -> bool:
        return self.config.enabled and self.queue.publish(snapshot)

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set();self.queue.close()
        if self._thread:self._thread.join(timeout_s)

    def run_foreground(self) -> None:
        """Run the native event loop on the calling thread (preferred on macOS)."""
        if not self.config.enabled:return
        self._run()

    def statistics(self) -> ViewerStatistics:
        elapsed=max(monotonic()-self._begin,1e-9)
        return ViewerStatistics(self.queue.published,self.queue.consumed,self.queue.dropped,
            self._rendered,self._malformed,self._rendered/elapsed,self._last_prepare_ms,len(self.queue),
            self.queue.closed or self._stop.is_set())

    def _run(self) -> None:
        last_geometry:PreparedGeometry|None=None
        try:
            if not self.renderer.open(self.controls):return
            period=1/self.config.target_render_hz
            while not self._stop.is_set():
                begin=perf_counter()
                if self.controls.paused and last_geometry is not None:
                    if not self.renderer.update(last_geometry,self.controls):break
                    sleep(period);continue
                snapshot=self.queue.newest(period)
                if snapshot is None:
                    if last_geometry is not None and not self.renderer.update(last_geometry,self.controls):break
                    continue
                try:geometry=prepare_snapshot(snapshot,self.config,self.controls)
                except (TypeError,ValueError):self._malformed+=1;continue
                self._last_prepare_ms=(perf_counter()-begin)*1000
                last_geometry=geometry
                if not self.renderer.update(geometry,self.controls):break
                self._rendered+=1;sleep(max(0,period-(perf_counter()-begin)))
        except Exception:
            # Visualization failure must never escape into autonomy code.
            self._malformed+=1
        finally:
            self.renderer.close();self.queue.close();self._stop.set()


def validate_snapshot(snapshot: VisualizationSnapshot) -> tuple[str, ...]:
    errors=[]
    if snapshot.schema_version!=SCHEMA_VERSION:errors.append("unsupported schema_version")
    if not np.isfinite(snapshot.timestamp_s) or snapshot.timestamp_s<0:errors.append("timestamp must be finite and non-negative")
    if snapshot.snapshot_id<0 or snapshot.map_revision<0:errors.append("IDs and revisions must be non-negative")
    point_fields=("stereo_camera_points","stereo_vehicle_points","tof_sensor_points",
        "tof_vehicle_points","combined_vehicle_points","sensor_origins_vehicle",
        "robot_footprint_vehicle","selected_path_vehicle","trajectory_vehicle",
        "traveled_trajectory_world","inspection_target_vehicle","occupancy_obstacles_vehicle",
        "changed_cells_vehicle")
    for name in point_fields:
        value=getattr(snapshot,name)
        if value is not None:
            array=np.asarray(value)
            if array.ndim!=2 or array.shape[1] not in (2,3):errors.append(f"{name} must be Nx2 or Nx3")
    for index,path in enumerate(snapshot.path_candidates_vehicle):
        array=np.asarray(path)
        if array.ndim!=2 or array.shape[1] not in (2,3):errors.append(f"path candidate {index} must be Nx2 or Nx3")
    if snapshot.stereo_colors_rgb is not None:
        colors=np.asarray(snapshot.stereo_colors_rgb)
        if colors.ndim!=2 or colors.shape[1]!=3:errors.append("stereo_colors_rgb must be Nx3")
        elif snapshot.stereo_vehicle_points is None or len(colors)!=len(snapshot.stereo_vehicle_points):errors.append("stereo colors length mismatch")
    if snapshot.confidence is not None:
        confidence=np.asarray(snapshot.confidence)
        reference=snapshot.combined_vehicle_points
        if confidence.ndim!=1 or reference is None or len(confidence)!=len(reference):errors.append("confidence must match combined points")
    return tuple(errors)


def prepare_snapshot(snapshot: VisualizationSnapshot, config: ViewerConfig,
                     controls: ViewerControls | None = None) -> PreparedGeometry:
    errors=validate_snapshot(snapshot)
    if errors:raise ValueError("; ".join(errors))
    controls=controls or ViewerControls(fixed_world=config.fixed_world,point_size=config.point_size)
    transform_pose=snapshot.estimated_pose or snapshot.vehicle_pose or Pose2D(0,0,0)
    use_world=controls.fixed_world
    points:dict[str,NDArray[np.float64]]={};colors:dict[str,NDArray[np.float64]]={};total=0
    sources=(("stereo",snapshot.stereo_vehicle_points,controls.show_stereo,(.15,.65,1.0),snapshot.stereo_colors_rgb,None),
             ("tof",snapshot.tof_vehicle_points,controls.show_tof,(1.0,.55,.1),None,None),
             ("combined",snapshot.combined_vehicle_points,controls.show_combined,(.75,.75,.75),None,snapshot.confidence),
             ("occupancy",snapshot.occupancy_obstacles_vehicle,controls.show_occupancy,(.85,.1,.1),None,None),
             ("changed",snapshot.changed_cells_vehicle,controls.show_occupancy,(1.0,0.,1.0),None,None))
    for name,source,visible,default_color,source_colors,confidence in sources:
        if source is None:continue
        total+=len(source)
        if not visible:continue
        cloud,indices=filter_and_downsample(source,config.voxel_size_m,config.max_points,
            confidence if controls.confidence_filter else None,config.confidence_threshold)
        if use_world:cloud=_transform_se2(cloud,transform_pose)
        points[name]=cloud
        if source_colors is not None:
            color_array=np.asarray(source_colors,float)[indices];colors[name]=np.clip(color_array,0,1)
        else:colors[name]=np.tile(default_color,(len(cloud),1))
    lines:dict[str,tuple[NDArray,np.ndarray,NDArray]]={}
    grid=_ground_grid(config.ground_grid_extent_m,config.ground_grid_spacing_m);lines["ground_grid"]=grid
    if controls.show_frames:
        axes_origin=np.array([transform_pose.x_m,transform_pose.y_m,0.]) if use_world else np.zeros(3)
        lines["vehicle_axes"]=_axes(axes_origin,transform_pose.heading_rad if use_world else 0.,.7)
        if snapshot.sensor_origins_vehicle is not None:
            for index,origin in enumerate(_as_xyz(snapshot.sensor_origins_vehicle)):
                world=_transform_se2(origin[None],transform_pose)[0] if use_world else origin
                lines[f"sensor_{index}_axes"]=_axes(world,transform_pose.heading_rad if use_world else 0.,.35)
        estimated=snapshot.estimated_pose or snapshot.vehicle_pose
        if estimated is not None:
            display=estimated if use_world else Pose2D(0,0,0)
            lines["estimated_pose"]=_pose_marker(display,(0.,1.,1.))
        if snapshot.ground_truth_pose is not None:
            display=snapshot.ground_truth_pose if use_world else _relative_pose(transform_pose,snapshot.ground_truth_pose)
            lines["ground_truth_pose"]=_pose_marker(display,(1.,1.,0.))
    if snapshot.robot_footprint_vehicle is not None:
        lines["footprint"]=_polyline(_world(snapshot.robot_footprint_vehicle,transform_pose,use_world),(0.,.2,0.))
    if controls.show_paths:
        for index,path in enumerate(snapshot.path_candidates_vehicle):lines[f"candidate_{index}"]=_polyline(_world(path,transform_pose,use_world),(.4,.4,.7))
        for name,path,color in (("selected_path",snapshot.selected_path_vehicle,(0.,1.,1.)),
                                ("trajectory",snapshot.trajectory_vehicle,(.1,1.,.1)),
                                ("traveled",snapshot.traveled_trajectory_world,(1.,1.,0.))):
            if path is not None:lines[name]=_polyline(_world(path,transform_pose,use_world and name!="traveled"),color)
        if snapshot.inspection_target_vehicle is not None:lines["inspection"]=_inspection(snapshot.inspection_target_vehicle,transform_pose,use_world)
    return PreparedGeometry(snapshot.snapshot_id,snapshot.timestamp_s,points,colors,lines,
        snapshot.warnings,snapshot.fault,total,sum(len(value) for value in points.values()))


def filter_and_downsample(points: NDArray[np.floating], voxel_size_m: float, max_points: int,
                          confidence: NDArray[np.floating] | None = None,
                          confidence_threshold: float = 0.) -> tuple[NDArray[np.float64],NDArray[np.int64]]:
    values=_as_xyz(points);mask=np.isfinite(values).all(1)
    if confidence is not None:
        conf=np.asarray(confidence,float)
        if conf.shape!=(len(values),):raise ValueError("confidence length mismatch")
        mask &= np.isfinite(conf)&(conf>=confidence_threshold)
    indices=np.flatnonzero(mask);values=values[indices]
    if len(values) and voxel_size_m>0:
        voxels=np.floor(values/voxel_size_m).astype(np.int64);_,first=np.unique(voxels,axis=0,return_index=True)
        first.sort();values=values[first];indices=indices[first]
    if len(values)>max_points:
        selection=np.linspace(0,len(values)-1,max_points,dtype=np.int64);values=values[selection];indices=indices[selection]
    return values,indices


def save_snapshot(snapshot: VisualizationSnapshot, path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True);arrays={};metadata={"schema_version":snapshot.schema_version,
        "timestamp_s":snapshot.timestamp_s,"snapshot_id":snapshot.snapshot_id,"map_revision":snapshot.map_revision,
        "warnings":snapshot.warnings,"fault":snapshot.fault}
    for name in VisualizationSnapshot.__dataclass_fields__:
        value=getattr(snapshot,name)
        if isinstance(value,np.ndarray):arrays[name]=value
        elif name=="path_candidates_vehicle":
            metadata["path_count"]=len(value)
            for index,array in enumerate(value):arrays[f"path_{index}"]=array
        elif isinstance(value,Pose2D):metadata[name]=[value.x_m,value.y_m,value.heading_rad]
        elif isinstance(value,RigidTransform):metadata[name]={"rotation":value.rotation.tolist(),"translation_m":value.translation_m.tolist()}
    arrays["metadata_json"]=np.asarray(json.dumps(metadata));np.savez_compressed(path,**arrays)


def load_snapshot(path: Path) -> VisualizationSnapshot:
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data["metadata_json"]));kwargs={key:data[key] for key in data.files if key not in {"metadata_json"} and not key.startswith("path_")}
        for name in ("vehicle_pose","estimated_pose","ground_truth_pose"):
            if name in metadata:kwargs[name]=Pose2D(*metadata[name])
        for name in ("camera_to_vehicle","tof_to_vehicle"):
            if name in metadata:kwargs[name]=RigidTransform(np.asarray(metadata[name]["rotation"]),np.asarray(metadata[name]["translation_m"]))
        kwargs["path_candidates_vehicle"]=tuple(data[f"path_{i}"] for i in range(metadata.get("path_count",0)))
        kwargs.update({key:metadata[key] for key in ("schema_version","timestamp_s","snapshot_id","map_revision","warnings","fault")})
        kwargs["warnings"]=tuple(kwargs["warnings"])
        return VisualizationSnapshot(**kwargs)


def render_prepared_preview(geometry: PreparedGeometry, path: Path,
                            size: tuple[int,int]=(1400,700)) -> None:
    """Save deterministic top/side projections for headless visual QA."""
    width,height=size;image=np.full((height,width,3),245,np.uint8)
    all_points=[value for value in geometry.points.values() if len(value)]
    for points,_,_ in geometry.lines.values():
        if len(points):all_points.append(points)
    values=np.vstack(all_points) if all_points else np.array([[-1,-1,-1],[1,1,1]],float)
    palettes={"stereo":(255,150,30),"tof":(30,140,255),"combined":(150,150,150),"occupancy":(20,20,210),"changed":(210,20,210)}
    for panel,(first,second),label in ((0,(0,1),"top: X right / Y forward"),(1,(1,2),"side: Y forward / Z up")):
        x0=panel*width//2;low=values[:,[first,second]].min(0);high=values[:,[first,second]].max(0);span=np.maximum(high-low,.1);low-=.08*span;high+=.08*span
        def px(points):
            p=points[:,[first,second]];return np.column_stack((x0+20+(p[:,0]-low[0])/(high[0]-low[0])*(width//2-40),height-20-(p[:,1]-low[1])/(high[1]-low[1])*(height-50))).astype(np.int32)
        cv2.rectangle(image,(x0,0),(x0+width//2-1,height-1),(190,190,190),1);cv2.putText(image,label,(x0+15,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
        for name,points in geometry.points.items():
            pixels=px(points)
            for point in pixels[::max(1,len(pixels)//20000)]:cv2.circle(image,tuple(point),1,palettes.get(name,(80,80,80)),-1)
        for name,(points,lines,colors) in geometry.lines.items():
            pixels=px(points)
            for line,color in zip(lines,colors):cv2.line(image,tuple(pixels[line[0]]),tuple(pixels[line[1]]),tuple(int(v*255) for v in color[::-1]),1,cv2.LINE_AA)
    cv2.putText(image,f"snapshot={geometry.snapshot_id} input={geometry.total_input_points} displayed={geometry.total_displayed_points}",(15,height-8),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
    path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),image)


class SnapshotReplay:
    def __init__(self, paths: list[Path], playback_speed: float = 1.) -> None:
        if playback_speed<=0:raise ValueError("playback_speed must be positive")
        loaded=[(load_snapshot(path),path) for path in paths];loaded.sort(key=lambda item:(item[0].timestamp_s,item[0].snapshot_id))
        self.paths=[path for _,path in loaded];self.index=0;self.paused=False;self.playback_speed=playback_speed
    @classmethod
    def from_folder(cls,folder:Path,playback_speed:float=1.):return cls(list(folder.glob("*.npz")),playback_speed)
    def current(self):return load_snapshot(self.paths[self.index]) if self.paths else None
    def next(self):
        if not self.paths:return None
        self.index=min(self.index+1,len(self.paths)-1);return self.current()
    def previous(self):
        if not self.paths:return None
        self.index=max(0,self.index-1);return self.current()
    def restart(self):self.index=0;return self.current()
    def toggle_pause(self):self.paused=not self.paused


class Open3DRenderer:
    """Lazy native renderer; importing the core package never imports Open3D."""
    def __init__(self,window_name:str="CarVision 3D",output_dir:Path=Path("outputs/viewer"),
                 *,diagnostics:bool=False,verification_screenshot:Path|None=None) -> None:
        self.window_name=window_name;self.output_dir=output_dir;self.diagnostics=diagnostics
        self.verification_screenshot=verification_screenshot;self._o3d=None;self.visualizer=None
        self.geometry={};self.loop_iterations=0;self._camera_initialized=False;self._last_view=None
    def open(self,controls:ViewerControls)->bool:
        try:import open3d as o3d
        except ImportError as error:raise RuntimeError("Open3D unavailable; install carvision[visualization] under Python 3.10-3.12") from error
        self._o3d=o3d;self.visualizer=o3d.visualization.VisualizerWithKeyCallback();
        if not self.visualizer.create_window(self.window_name,width=1280,height=800):return False
        self._register_controls(controls);options=self.visualizer.get_render_option();options.point_size=max(3.,controls.point_size)
        options.background_color=np.array([.025,.035,.055]);return True
    def update(self,prepared:PreparedGeometry,controls:ViewerControls)->bool:
        o3d=self._o3d;active=set();self.loop_iterations+=1;added=[];updated=[]
        for name,points in prepared.points.items():
            active.add(name);geometry=self.geometry.get(name)
            if geometry is None:
                geometry=o3d.geometry.PointCloud();geometry.points=o3d.utility.Vector3dVector(points);geometry.colors=o3d.utility.Vector3dVector(prepared.colors[name])
                self.geometry[name]=geometry
                if not self.visualizer.add_geometry(geometry,reset_bounding_box=False):raise RuntimeError(f"Open3D failed to add point geometry {name}")
                added.append(name)
            else:
                geometry.points=o3d.utility.Vector3dVector(points);geometry.colors=o3d.utility.Vector3dVector(prepared.colors[name]);self.visualizer.update_geometry(geometry);updated.append(name)
        for name,(points,lines,colors) in prepared.lines.items():
            if len(points)==0 or len(lines)==0:continue
            key=f"line:{name}";active.add(key);geometry=self.geometry.get(key)
            if geometry is None:
                geometry=o3d.geometry.LineSet();geometry.points=o3d.utility.Vector3dVector(points);geometry.lines=o3d.utility.Vector2iVector(lines);geometry.colors=o3d.utility.Vector3dVector(colors);self.geometry[key]=geometry
                if not self.visualizer.add_geometry(geometry,reset_bounding_box=False):raise RuntimeError(f"Open3D failed to add line geometry {name}")
                added.append(key)
            else:
                geometry.points=o3d.utility.Vector3dVector(points);geometry.lines=o3d.utility.Vector2iVector(lines);geometry.colors=o3d.utility.Vector3dVector(colors);self.visualizer.update_geometry(geometry);updated.append(key)
        inactive=[]
        for name,geometry in self.geometry.items():
            if name not in active:
                self.visualizer.remove_geometry(geometry,reset_bounding_box=False);inactive.append(name)
        for name in inactive:del self.geometry[name]
        self.visualizer.get_render_option().point_size=max(3.,controls.point_size)
        if not self._camera_initialized and self.geometry:
            self.visualizer.reset_view_point(True);self._initialize_camera(prepared,controls);self._camera_initialized=True
        elif controls.requested_view!=self._last_view:self._initialize_camera(prepared,controls)
        if controls.save_cloud_requested:self._save_cloud(prepared);controls.save_cloud_requested=False
        alive=bool(self.visualizer.poll_events())
        if not alive:
            if self.diagnostics:print(f"viewer loop={self.loop_iterations}: poll_events returned false",flush=True)
            return False
        self.visualizer.update_renderer()
        screenshot=None
        if controls.screenshot_requested:self.output_dir.mkdir(parents=True,exist_ok=True);screenshot=self.output_dir/f"frame_{prepared.snapshot_id:06d}.png";controls.screenshot_requested=False
        elif self.verification_screenshot is not None and self.loop_iterations==5:screenshot=self.verification_screenshot
        if screenshot is not None:screenshot.parent.mkdir(parents=True,exist_ok=True);self.visualizer.capture_screen_image(str(screenshot),do_render=True)
        if self.diagnostics and (self.loop_iterations<=5 or self.loop_iterations%30==0 or added):self._print_diagnostics(prepared,added,updated,controls)
        return True
    def close(self)->None:
        if self.visualizer:self.visualizer.destroy_window();self.visualizer=None
    def _register_controls(self,c):
        v=self.visualizer
        for key,name in ((ord(" "),"paused"),(ord("1"),"show_stereo"),(ord("2"),"show_tof"),(ord("3"),"show_combined"),(ord("F"),"show_frames"),(ord("P"),"show_paths"),(ord("O"),"show_occupancy"),(ord("C"),"confidence_filter"),(ord("M"),"fixed_world")):
            v.register_key_callback(key,lambda vis,n=name:(c.toggle(n),False)[1])
        for key,view in ((ord("R"),"perspective"),(ord("T"),"top"),(ord("S"),"side"),(ord("V"),"front")):v.register_key_callback(key,lambda vis,name=view:(setattr(c,"requested_view",name),False)[1])
        v.register_key_callback(ord("+"),lambda vis:(setattr(c,"point_size",min(c.point_size+1,10)),False)[1]);v.register_key_callback(ord("-"),lambda vis:(setattr(c,"point_size",max(c.point_size-1,1)),False)[1])
        v.register_key_callback(ord("K"),lambda vis:(setattr(c,"save_cloud_requested",True),False)[1]);v.register_key_callback(ord("I"),lambda vis:(setattr(c,"screenshot_requested",True),False)[1])
        v.register_key_callback(ord("N"),lambda vis:(setattr(c,"replay_next_requested",True),False)[1]);v.register_key_callback(ord("B"),lambda vis:(setattr(c,"replay_previous_requested",True),False)[1]);v.register_key_callback(ord("H"),lambda vis:(setattr(c,"replay_restart_requested",True),False)[1])
    def _initialize_camera(self,prepared,c):
        values=[]
        values.extend(points for points in prepared.points.values() if len(points))
        values.extend(points for name,(points,lines,_) in prepared.lines.items() if name!="ground_grid" and len(lines))
        if not values:values=[prepared.lines["ground_grid"][0]]
        scene=np.vstack(values);low=scene.min(0);high=scene.max(0);center=(low+high)/2
        views={"top":((0,0,1),(0,1,0)),"side":((1,0,0),(0,0,1)),"front":((0,-1,0),(0,0,1)),"perspective":((.55,-.7,.45),(0,0,1))}
        front,up=views[c.requested_view];control=self.visualizer.get_view_control();control.set_lookat(center);control.set_front(front);control.set_up(up);control.set_zoom(.65);self._last_view=c.requested_view
        if self.diagnostics:print(f"camera look_at={center.tolist()} front={front} up={up} zoom=0.65 scene_min={low.tolist()} scene_max={high.tolist()}",flush=True)
    def _print_diagnostics(self,p,added,updated,c):
        counts={name:len(points) for name,points in p.points.items()};line_counts={name:(len(points),len(lines)) for name,(points,lines,_) in p.lines.items()}
        bounds={name:(points.min(0).tolist(),points.max(0).tolist()) for name,points in p.points.items() if len(points)}
        print(f"renderer loop={self.loop_iterations} snapshot={p.snapshot_id} point_counts={counts} line_counts={line_counts} prepared_total={p.total_displayed_points} bounds={bounds} added={added} updated={updated} view={c.requested_view}",flush=True)
    def _save_cloud(self,p):
        clouds=[value for value in p.points.values() if len(value)]
        if not clouds:return
        self.output_dir.mkdir(parents=True,exist_ok=True);cloud=self._o3d.geometry.PointCloud();cloud.points=self._o3d.utility.Vector3dVector(np.vstack(clouds));self._o3d.io.write_point_cloud(str(self.output_dir/f"frame_{p.snapshot_id:06d}.ply"),cloud)


def run_open3d_smoke_test(screenshot:Path,seconds:float=4.,diagnostics:bool=True)->bool:
    """Draw native primitives without the snapshot queue and save proof pixels."""
    import open3d as o3d
    visualizer=o3d.visualization.Visualizer();
    if not visualizer.create_window("CarVision Open3D smoke test",width=960,height=640):return False
    options=visualizer.get_render_option();options.background_color=np.array([.02,.03,.05]);options.point_size=12.
    axes=o3d.geometry.TriangleMesh.create_coordinate_frame(size=.8,origin=[0,0,0]);visualizer.add_geometry(axes)
    grid_points=[];grid_lines=[]
    for value in np.arange(-3,3.1,.5):
        start=len(grid_points);grid_points.extend([[-3,value,0],[3,value,0],[value,-3,0],[value,3,0]]);grid_lines.extend([[start,start+1],[start+2,start+3]])
    grid=o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.asarray(grid_points)),o3d.utility.Vector2iVector(np.asarray(grid_lines)));grid.colors=o3d.utility.Vector3dVector(np.tile([.35,.35,.35],(len(grid_lines),1)));visualizer.add_geometry(grid)
    points=o3d.geometry.PointCloud();points.points=o3d.utility.Vector3dVector(np.array([[-1,1,.35],[0,2,.65],[1,3,1.0]]));points.colors=o3d.utility.Vector3dVector(np.eye(3));visualizer.add_geometry(points)
    for center,color in (([-1,1,.15],[1,0,0]),([0,2,.15],[0,1,0]),([1,3,.15],[0,0,1])):
        box=o3d.geometry.TriangleMesh.create_box(.3,.3,.3);box.translate(np.asarray(center)-.15);box.paint_uniform_color(color);visualizer.add_geometry(box)
    visualizer.reset_view_point(True);control=visualizer.get_view_control();control.set_lookat([0,1.5,.3]);control.set_front([.55,-.7,.45]);control.set_up([0,0,1]);control.set_zoom(.7)
    if diagnostics:print("smoke geometry: axes=1 grid_lines=52 colored_points=3 boxes=3 camera look_at=[0,1.5,0.3] front=[0.55,-0.7,0.45] up=[0,0,1] zoom=0.7",flush=True)
    begin=monotonic();iterations=0;alive=True
    while alive and monotonic()-begin<seconds:
        alive=bool(visualizer.poll_events());visualizer.update_renderer();iterations+=1;sleep(.01)
        if iterations==10:screenshot.parent.mkdir(parents=True,exist_ok=True);visualizer.capture_screen_image(str(screenshot),do_render=True)
    if diagnostics:print(f"smoke loop_iterations={iterations} poll_events_alive={alive} screenshot={screenshot}",flush=True)
    visualizer.destroy_window();return screenshot.exists()


def _as_xyz(points):
    values=np.asarray(points,float)
    if values.ndim!=2 or values.shape[1] not in (2,3):raise ValueError("points must be Nx2 or Nx3")
    return np.column_stack((values,np.zeros(len(values)))) if values.shape[1]==2 else values
def _transform_se2(points,pose):
    values=_as_xyz(points);c=np.cos(pose.heading_rad);s=np.sin(pose.heading_rad);result=values.copy();result[:,0]=pose.x_m+c*values[:,0]+s*values[:,1];result[:,1]=pose.y_m-s*values[:,0]+c*values[:,1];return result
def _world(points,pose,enabled):return _transform_se2(points,pose) if enabled else _as_xyz(points)
def _polyline(points,color):
    values=_as_xyz(points);lines=np.column_stack((np.arange(max(0,len(values)-1)),np.arange(1,len(values)))).astype(np.int32);return values,lines,np.tile(color,(len(lines),1))
def _axes(origin,heading,size):
    c=np.cos(heading);s=np.sin(heading);directions=np.array([[c,-s,0],[s,c,0],[0,0,1.]])*size;points=np.vstack((origin,origin+directions));lines=np.array([[0,1],[0,2],[0,3]],np.int32);return points,lines,np.eye(3)
def _ground_grid(extent,spacing):
    coordinates=np.arange(-extent,extent+spacing/2,spacing);points=[];lines=[]
    for value in coordinates:
        start=len(points);points.extend([[-extent,value,0],[extent,value,0],[value,-extent,0],[value,extent,0]]);lines.extend([[start,start+1],[start+2,start+3]])
    return np.asarray(points,float),np.asarray(lines,np.int32),np.tile((.3,.3,.3),(len(lines),1))
def _inspection(target,pose,world):
    values=_as_xyz(target);origin=np.zeros((1,3));points=np.vstack((origin,values[:1]));points=_world(points,pose,world);return points,np.array([[0,1]],np.int32),np.array([[1.,0.,1.]])
def _pose_marker(pose,color):
    origin=np.array([pose.x_m,pose.y_m,.1]);forward=np.array([np.sin(pose.heading_rad),np.cos(pose.heading_rad),0.]);right=np.array([np.cos(pose.heading_rad),-np.sin(pose.heading_rad),0.]);points=np.vstack((origin,origin+.45*forward,origin+.18*right));lines=np.array([[0,1],[0,2]],np.int32);return points,lines,np.tile(color,(2,1))
def _relative_pose(reference,target):
    dx=target.x_m-reference.x_m;dy=target.y_m-reference.y_m;c=np.cos(reference.heading_rad);s=np.sin(reference.heading_rad);return Pose2D(c*dx-s*dy,s*dx+c*dy,target.heading_rad-reference.heading_rad)
