# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive editor for FrameTransformer offsets and scene prim poses.

FrameTransformer offsets are not USD prims, so they cannot be dragged with
Isaac Sim's manipulator. This module materializes each named frame as a
child Xform (RGB axes) under its parent rigid body. While the timeline is
paused, those gizmos are selectable: use W/E in the viewport, or the
keyboard nudges below, then print copy-pasteable ``OffsetCfg`` snippets.

Keyboard (viewport must be focused):
    N / B         next / previous frame
    I/K J/L U/O   translate +/- X/Y/Z (parent frame)
    Shift+those   rotate about that axis
    [ / ]         finer / coarser step
    T             set selected frame from current TCP
    P             print all offsets
    H             print this help
    0             restore selected frame to the value loaded from the scene
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

import carb
import omni.appwindow
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.sensors import FrameTransformer

HELP_TEXT = """HiveBoard pose editor
  Mouse: select a PoseEditor_* prim in the viewport or Stage, then
         W = translate, E = rotate (Isaac Sim gizmos).
  N/B         cycle frames
  I/K J/L U/O translate X/Y/Z
  Shift+axis  rotate about that axis
  [ / ]       step size
  T           snap selected frame to current TCP
  P           print OffsetCfg snippets
  0           restore selected frame
  H           help
"""


def _fmt_tuple(values: tuple[float, ...] | np.ndarray, digits: int = 8) -> str:
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def _pos_quat_from_matrix(matrix: Gf.Matrix4d) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    pos = (float(translation[0]), float(translation[1]), float(translation[2]))
    rot = (
        float(quat.GetReal()),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
    )
    return pos, _normalize_quat(rot)


def _normalize_quat(rot: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    array = np.asarray(rot, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm < 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    array = array / norm
    if array[0] < 0.0:
        array = -array
    return (float(array[0]), float(array[1]), float(array[2]), float(array[3]))


def _matrix_from_pos_quat(
    pos: tuple[float, float, float], rot: tuple[float, float, float, float]
) -> Gf.Matrix4d:
    quat = Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3])))
    matrix = Gf.Matrix4d()
    matrix.SetRotate(quat)
    matrix.SetTranslateOnly(Gf.Vec3d(*pos))
    return matrix


def _rpy_deg(rot: tuple[float, float, float, float]) -> tuple[float, float, float]:
    euler = Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]]).as_euler("xyz", degrees=True)
    return (float(euler[0]), float(euler[1]), float(euler[2]))


def _offset_snippet(pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> str:
    return (
        "OffsetCfg(\n"
        f"    pos={_fmt_tuple(pos)},\n"
        f"    rot={_fmt_tuple(rot)},\n"
        ")"
    )


def _define_preview_material(stage: Usd.Stage, path: str, rgb: tuple[float, float, float]) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _spawn_axis_gizmo(stage: Usd.Stage, prim_path: str, axis_length: float = 0.10) -> Usd.Prim:
    """Create a pickable RGB-axis xform with no physics."""
    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.MakeMatrixXform()
    radius = axis_length * 0.045
    cone_height = axis_length * 0.22
    cone_radius = radius * 2.2
    materials_root = f"{prim_path}/Looks"
    materials = {
        "x": _define_preview_material(stage, f"{materials_root}/Red", (0.85, 0.12, 0.12)),
        "y": _define_preview_material(stage, f"{materials_root}/Green", (0.12, 0.75, 0.18)),
        "z": _define_preview_material(stage, f"{materials_root}/Blue", (0.15, 0.35, 0.90)),
        "origin": _define_preview_material(stage, f"{materials_root}/White", (0.92, 0.92, 0.92)),
    }

    sphere = UsdGeom.Sphere.Define(stage, f"{prim_path}/origin")
    sphere.CreateRadiusAttr(axis_length * 0.08)
    _bind_material(sphere.GetPrim(), materials["origin"])

    def _axis(name: str, rotate_xyz_deg: tuple[float, float, float]) -> None:
        axis_xf = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}")
        axis_xf.AddRotateXYZOp().Set(Gf.Vec3f(*rotate_xyz_deg))
        shaft = UsdGeom.Cylinder.Define(stage, f"{prim_path}/{name}/shaft")
        shaft.CreateRadiusAttr(radius)
        shaft.CreateHeightAttr(axis_length)
        shaft.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, axis_length * 0.5))
        _bind_material(shaft.GetPrim(), materials[name])
        cone = UsdGeom.Cone.Define(stage, f"{prim_path}/{name}/tip")
        cone.CreateRadiusAttr(cone_radius)
        cone.CreateHeightAttr(cone_height)
        cone.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, axis_length + cone_height * 0.5))
        _bind_material(cone.GetPrim(), materials[name])

    # USD cylinders/cones are +Z. Rotate so X/Y match the local frame.
    _axis("x", (0.0, 90.0, 0.0))
    _axis("y", (-90.0, 0.0, 0.0))
    _axis("z", (0.0, 0.0, 0.0))
    return xform.GetPrim()


def _set_local_pose(
    prim: Usd.Prim, pos: tuple[float, float, float], rot: tuple[float, float, float, float]
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*pos))
    orient_op.Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3]))))


def _get_local_pose(prim: Usd.Prim) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
    return _pos_quat_from_matrix(matrix)


def _get_world_matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def _resolve_env_prim_path(path: str, env_ns: str) -> str:
    return path.replace("{ENV_REGEX_NS}", env_ns)


@dataclass
class EditableFrame:
    """One named FrameTransformer target that can be dragged in the viewport."""

    sensor_name: str
    frame_name: str
    parent_prim_path: str
    gizmo_prim_path: str
    original_pos: tuple[float, float, float]
    original_rot: tuple[float, float, float, float]


@dataclass
class PoseEditor:
    """Runtime controller for in-sim pose authoring."""

    env: object
    env_index: int = 0
    axis_length: float = 0.10
    frames: list[EditableFrame] = field(default_factory=list)
    selected: int = 0
    pos_step: float = 0.005
    rot_step_deg: float = 2.0
    _tcp_world: Gf.Matrix4d | None = field(default=None, init=False, repr=False)
    _keyboard_sub: object | None = field(default=None, init=False, repr=False)
    _window: object | None = field(default=None, init=False, repr=False)
    _status_label: object | None = field(default=None, init=False, repr=False)
    _list_label: object | None = field(default=None, init=False, repr=False)

    def setup(self) -> None:
        """Spawn gizmos, cache TCP, pause physics, and bind input."""
        self._cache_tcp()
        self._spawn_frame_gizmos()
        if not self.frames:
            raise RuntimeError("No FrameTransformer target frames were found in the scene.")
        self._pause_timeline()
        self._select_index(0)
        self._bind_keyboard()
        self._build_window()
        print(HELP_TEXT)
        print(self.format_all())

    def close(self) -> None:
        if self._keyboard_sub is not None:
            input_iface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            input_iface.unsubscribe_to_keyboard_events(keyboard, self._keyboard_sub)
            self._keyboard_sub = None
        if self._window is not None:
            self._window = None

    def update(self) -> None:
        """Sync UI from USD (picks up mouse-gizmo edits)."""
        self._sync_selection_from_stage()
        self._refresh_window()

    def current(self) -> EditableFrame:
        return self.frames[self.selected]

    def format_all(self) -> str:
        lines = ["# Copy-paste into the task scene.py FrameTransformerCfg"]
        stage = omni.usd.get_context().get_stage()
        for item in self.frames:
            prim = stage.GetPrimAtPath(item.gizmo_prim_path)
            pos, rot = _get_local_pose(prim)
            rpy = _rpy_deg(rot)
            lines.append(
                f"# {item.sensor_name}.{item.frame_name}  parent={item.parent_prim_path}"
            )
            lines.append(f"# rpy_xyz_deg={_fmt_tuple(rpy, digits=2)}")
            lines.append(_offset_snippet(pos, rot))
            lines.append("")
        return "\n".join(lines)

    def print_all(self) -> None:
        print(self.format_all())

    def snap_selected_to_tcp(self) -> None:
        if self._tcp_world is None:
            print("[WARN] No TCP pose was cached; cannot snap.")
            return
        item = self.current()
        stage = omni.usd.get_context().get_stage()
        parent = stage.GetPrimAtPath(item.parent_prim_path)
        if not parent.IsValid():
            print(f"[WARN] Parent prim missing: {item.parent_prim_path}")
            return
        parent_world = _get_world_matrix(parent)
        offset = parent_world.GetInverse() * self._tcp_world
        pos, rot = _pos_quat_from_matrix(offset)
        gizmo = stage.GetPrimAtPath(item.gizmo_prim_path)
        _set_local_pose(gizmo, pos, rot)
        print(f"[INFO] Snapped {item.sensor_name}.{item.frame_name} to TCP")
        print(_offset_snippet(pos, rot))

    def restore_selected(self) -> None:
        item = self.current()
        stage = omni.usd.get_context().get_stage()
        gizmo = stage.GetPrimAtPath(item.gizmo_prim_path)
        _set_local_pose(gizmo, item.original_pos, item.original_rot)
        print(f"[INFO] Restored {item.sensor_name}.{item.frame_name}")

    def dump_stage_selection(self) -> None:
        """Print local and world pose of the prim currently selected in the Stage."""
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not paths:
            print("[INFO] Nothing selected in the Stage.")
            return
        stage = omni.usd.get_context().get_stage()
        env_ns = self.env.scene.env_prim_paths[self.env_index]
        env_prim = stage.GetPrimAtPath(env_ns)
        env_world = _get_world_matrix(env_prim) if env_prim.IsValid() else Gf.Matrix4d(1.0)
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                continue
            local_pos, local_rot = _get_local_pose(prim)
            world_pos, world_rot = _pos_quat_from_matrix(_get_world_matrix(prim))
            env_local = env_world.GetInverse() * _get_world_matrix(prim)
            env_pos, env_rot = _pos_quat_from_matrix(env_local)
            print(f"# prim {path}")
            print(f"# local (parent frame)\n{_offset_snippet(local_pos, local_rot)}")
            print(f"# env origin {env_ns}\n{_offset_snippet(env_pos, env_rot)}")
            print(f"# world\n{_offset_snippet(world_pos, world_rot)}")
            print("")

    def _cache_tcp(self) -> None:
        sensors = self.env.scene.sensors
        if "ee_frame" not in sensors:
            print("[WARN] Scene has no ee_frame; TCP snap is disabled.")
            return
        ee: FrameTransformer = sensors["ee_frame"]
        if ee.data.target_pos_w.numel() == 0:
            print("[WARN] ee_frame has no target pose; TCP snap is disabled.")
            return
        pos = ee.data.target_pos_w[self.env_index, 0].detach().cpu().numpy()
        rot = ee.data.target_quat_w[self.env_index, 0].detach().cpu().numpy()
        self._tcp_world = _matrix_from_pos_quat(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            (float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])),
        )

    def _spawn_frame_gizmos(self) -> None:
        stage = omni.usd.get_context().get_stage()
        env_ns = self.env.scene.env_prim_paths[self.env_index]
        sensors = self.env.scene.sensors
        for sensor_name, sensor in sensors.items():
            if not isinstance(sensor, FrameTransformer):
                continue
            for frame_cfg in sensor.cfg.target_frames:
                frame_name = frame_cfg.name or frame_cfg.prim_path.rsplit("/", 1)[-1]
                parent_path = _resolve_env_prim_path(frame_cfg.prim_path, env_ns)
                parent = stage.GetPrimAtPath(parent_path)
                if not parent.IsValid():
                    print(f"[WARN] Skipping {sensor_name}.{frame_name}: missing {parent_path}")
                    continue
                gizmo_path = f"{parent_path}/PoseEditor_{frame_name}"
                if stage.GetPrimAtPath(gizmo_path).IsValid():
                    stage.RemovePrim(gizmo_path)
                _spawn_axis_gizmo(stage, gizmo_path, axis_length=self.axis_length)
                pos = tuple(float(v) for v in frame_cfg.offset.pos)
                rot = _normalize_quat(tuple(float(v) for v in frame_cfg.offset.rot))
                _set_local_pose(stage.GetPrimAtPath(gizmo_path), pos, rot)
                self.frames.append(
                    EditableFrame(
                        sensor_name=sensor_name,
                        frame_name=frame_name,
                        parent_prim_path=parent_path,
                        gizmo_prim_path=gizmo_path,
                        original_pos=pos,
                        original_rot=rot,
                    )
                )

    def _pause_timeline(self) -> None:
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        sim = getattr(self.env, "sim", None)
        if sim is not None and hasattr(sim, "pause"):
            sim.pause()
        print("[INFO] Timeline paused. Drag PoseEditor gizmos; physics will not fight you.")

    def _select_index(self, index: int) -> None:
        self.selected = int(index) % len(self.frames)
        path = self.current().gizmo_prim_path
        omni.usd.get_context().get_selection().set_selected_prim_paths([path], True)
        item = self.current()
        print(f"[INFO] Selected {item.sensor_name}.{item.frame_name}")

    def _sync_selection_from_stage(self) -> None:
        paths = set(omni.usd.get_context().get_selection().get_selected_prim_paths())
        if not paths:
            return
        for index, item in enumerate(self.frames):
            if item.gizmo_prim_path in paths or any(
                selected.startswith(item.gizmo_prim_path + "/") for selected in paths
            ):
                if index != self.selected:
                    self.selected = index
                return

    def _nudge(self, translation: np.ndarray, rpy_deg: np.ndarray) -> None:
        item = self.current()
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(item.gizmo_prim_path)
        pos, rot = _get_local_pose(prim)
        new_pos = (
            pos[0] + float(translation[0]),
            pos[1] + float(translation[1]),
            pos[2] + float(translation[2]),
        )
        current = Rotation.from_quat([rot[1], rot[2], rot[3], rot[0]])
        delta = Rotation.from_euler("xyz", rpy_deg, degrees=True)
        combined = current * delta
        quat_xyzw = combined.as_quat()
        new_rot = _normalize_quat(
            (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
        )
        _set_local_pose(prim, new_pos, new_rot)

    def _bind_keyboard(self) -> None:
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        input_iface = carb.input.acquire_input_interface()
        self._keyboard_sub = input_iface.subscribe_to_keyboard_events(
            keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event),
        )

    def _shift_down(self) -> bool:
        try:
            input_iface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            return bool(
                input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.LEFT_SHIFT)
                or input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.RIGHT_SHIFT)
            )
        except (AttributeError, RuntimeError):
            return False

    def _on_keyboard_event(self, event) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name
        if name == "N":
            self._select_index(self.selected + 1)
        elif name == "B":
            self._select_index(self.selected - 1)
        elif name == "P":
            self.print_all()
        elif name == "T":
            self.snap_selected_to_tcp()
        elif name == "H":
            print(HELP_TEXT)
        elif name in {"KEY_0", "NUMBER_0", "NUMPAD_0"}:
            self.restore_selected()
        elif name == "LEFT_BRACKET":
            self.pos_step = max(0.0005, self.pos_step * 0.5)
            self.rot_step_deg = max(0.25, self.rot_step_deg * 0.5)
            print(f"[INFO] step pos={self.pos_step:.4f} m  rot={self.rot_step_deg:.2f} deg")
        elif name == "RIGHT_BRACKET":
            self.pos_step = min(0.1, self.pos_step * 2.0)
            self.rot_step_deg = min(45.0, self.rot_step_deg * 2.0)
            print(f"[INFO] step pos={self.pos_step:.4f} m  rot={self.rot_step_deg:.2f} deg")
        elif name in {"I", "K", "J", "L", "U", "O"}:
            axis = {"I": 0, "K": 0, "J": 1, "L": 1, "U": 2, "O": 2}[name]
            sign = 1.0 if name in {"I", "J", "U"} else -1.0
            translation = np.zeros(3)
            rpy = np.zeros(3)
            if self._shift_down():
                rpy[axis] = sign * self.rot_step_deg
            else:
                translation[axis] = sign * self.pos_step
            self._nudge(translation, rpy)
        return True

    def _build_window(self) -> None:
        try:
            import omni.ui as ui
        except ImportError:
            return

        self._window = ui.Window("HiveBoard Pose Editor", width=440, height=420)
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Select a PoseEditor_* axis in the viewport, then W/E to drag.")
                self._list_label = ui.Label("", word_wrap=True)
                self._status_label = ui.Label("", word_wrap=True)
                with ui.HStack(height=28):
                    ui.Button("Previous", clicked_fn=lambda: self._select_index(self.selected - 1))
                    ui.Button("Next", clicked_fn=lambda: self._select_index(self.selected + 1))
                with ui.HStack(height=28):
                    ui.Button("Print OffsetCfg", clicked_fn=self.print_all)
                    ui.Button("Snap to TCP", clicked_fn=self.snap_selected_to_tcp)
                with ui.HStack(height=28):
                    ui.Button("Restore selected", clicked_fn=self.restore_selected)
                    ui.Button("Dump Stage selection", clicked_fn=self.dump_stage_selection)
        self._refresh_window()

    def _refresh_window(self) -> None:
        if self._list_label is None or self._status_label is None:
            return
        names = []
        for index, item in enumerate(self.frames):
            mark = ">" if index == self.selected else " "
            names.append(f"{mark} {item.sensor_name}.{item.frame_name}")
        self._list_label.text = "\n".join(names)
        stage = omni.usd.get_context().get_stage()
        item = self.current()
        prim = stage.GetPrimAtPath(item.gizmo_prim_path)
        pos, rot = _get_local_pose(prim)
        rpy = _rpy_deg(rot)
        self._status_label.text = (
            f"{item.sensor_name}.{item.frame_name}\n"
            f"pos {_fmt_tuple(pos, digits=4)}\n"
            f"quat wxyz {_fmt_tuple(rot, digits=4)}\n"
            f"rpy xyz deg {_fmt_tuple(rpy, digits=1)}\n"
            f"step {self.pos_step:.4f} m / {self.rot_step_deg:.2f} deg"
        )


def enable_frame_debug_vis(env_cfg) -> None:
    """Turn on FrameTransformer markers so the authored frames stay visible."""
    scene_cfg = env_cfg.scene
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        entity_cfg = getattr(scene_cfg, name)
        if hasattr(entity_cfg, "debug_vis") and hasattr(entity_cfg, "target_frames"):
            entity_cfg.debug_vis = True
    commands_cfg = getattr(env_cfg, "commands", None)
    pose_command = getattr(commands_cfg, "pose_command", None) if commands_cfg is not None else None
    if pose_command is not None and hasattr(pose_command, "debug_vis"):
        pose_command.debug_vis = True
