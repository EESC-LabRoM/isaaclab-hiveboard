#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drag command goals, rotation references and the tool offset in Newton/Viser.

    uv run python scripts/command_edit.py
    uv run python scripts/command_edit.py --setup logs/command_setup.json

This is a kinematic authoring preview. Tick **cuRobo plan** on a GoTo/Rotate, or
insert a Curobo command, to save CuroboPlanned* terms. play.py then executes
those joint plans; the editor itself never launches cuRobo.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import queue
import sys
import time
from pathlib import Path

# Avoid concurrent USD authoring races in the kitless scene importer. Respect an
# explicit user setting. This must precede imports that initialize OpenUSD/TBB.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import gymnasium as gym
import isaaclab_hiveboard  # noqa: F401
import numpy as np
import torch
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    SequentialPoseCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.utils.command_preview import (
    PreviewIK,
    build_segments,
    numpy,
    pose_to_viser,
    refresh_frame_sensors,
    tensor,
    viser_to_pose,
)
from isaaclab_hiveboard.utils.command_setup import (
    apply_setup,
    as_curobo_command,
    as_direct_command,
    is_curobo_command,
    load_setup,
    make_setup,
    planner_settings,
    save_setup,
    validate_command,
    validate_setup,
)

import isaaclab.utils.math as math_utils

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

DEFAULT_TASK = "Isaac-HiveBoard-Spot-BallValve-Play-v0"
COMMAND_LABELS = {
    "CuroboPlannedGoToFrame": "cuRobo GoTo",
    "CuroboPlannedRotateFrame": "cuRobo Rotate",
    "GoToFrame": "GoTo",
    "RotateFrame": "Rotate",
    "ScrewFrame": "Screw",
}


class _PreviewCommand(SequentialPoseCommand):
    """Keep reset from launching cuRobo plans before the editor opens."""

    def _resample_command(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        self._sample_valve_task(ids)
        pos, quat = self._get_ee_in_base_frame(ids)
        self._command[:, 1:4] = pos
        self._command[:, 4:8] = quat


class CommandEditor:
    def __init__(self, env, viewer, task: str, out: Path, ik_iters: int):
        self.env, self.viewer, self.task, self.out = env, viewer, task, out
        # Newton currently exposes its Viser server through this single private attribute.
        self.server = viewer._server
        self.term = env.command_manager.get_term("pose_command")
        self.commands = copy.deepcopy(list(self.term.cfg.commands))
        self.ik = PreviewIK(env, self.term)
        self.ik.offset = self.ik.offset or SequentialPoseCommandCfg.OffsetCfg()
        self.ik_iters = ik_iters
        self.selected, self.fraction = 0, 1.0
        self.playing, self.dirty = False, False
        self.pending = queue.SimpleQueue()
        self.epoch = 0
        self.panel = None
        self.gizmo_poses = {}
        self.markers = []
        self.reference_options = {}
        for sensor_name, sensor in env.scene.sensors.items():
            if sensor_name == "ee_frame":
                continue
            for frame in getattr(sensor.cfg, "target_frames", ()):
                self.reference_options[f"{sensor_name}/{frame.name}"] = (sensor_name, frame.name)
        self._create_ui()
        self.rebuild()
        self.show_selected()

    def _enqueue(self, callback, *, epoch=None):
        def wrapped(event):
            if getattr(event, "client", None) is not None:
                # Capture values on the Viser thread; execute all simulation work in poll().
                value = copy.deepcopy(getattr(event.target, "value", None))
                self.pending.put((epoch, callback, value))

        return wrapped

    def _button(self, label, callback):
        handle = self.server.gui.add_button(label)
        handle.on_click(self._enqueue(lambda _: callback()))
        return handle

    def _create_ui(self):
        gui = self.server.gui
        gui.add_markdown(
            "## HiveBoard command setup\n"
            "Drag a goal or rotation reference. RGB axes are X/Y/Z. "
            "**Kinematic preview**: objects stay at their reset poses. "
            "Enable **cuRobo plan** to save planned joint commands for play.py.",
            order=0,
        )
        self.status = gui.add_markdown("Loading…", order=1)
        with gui.add_folder("Sequence", order=2):
            self.selection = gui.add_dropdown("Command", options=["Loading"])
            self.selection.on_update(self._enqueue(lambda value: self.select(int(value.split(":", 1)[0]))))
            self.scrub = gui.add_slider("Progress", min=0.0, max=1.0, step=0.01, initial_value=1.0)
            self.scrub.on_update(self._enqueue(self.seek))
            self._button("Preview sequence", self.play)
            self._button("Pause", lambda: setattr(self, "playing", False))
            self._button("Reset robot pose", self.reset)
            with gui.add_folder("Edit sequence", expand_by_default=False):
                self._button("Duplicate command", self.duplicate)
                self._button("Move earlier", lambda: self.move(-1))
                self._button("Move later", lambda: self.move(1))
                self._button("Delete command", self.delete)
                self.add_kind = gui.add_dropdown("New command", options=self._insert_kinds())
                self._button("Insert after selected", self.insert)
        with gui.add_folder("Tool center point (TCP)", order=4, expand_by_default=False):
            gui.add_markdown(
                f"Offset relative to **{self.term.cfg.body_name}**. Rotations below are XYZ Euler degrees."
            )
            self.calibrate = gui.add_checkbox("Drag TCP offset", initial_value=False)
            self.calibrate.on_update(self._enqueue(self.set_calibrating))
            self.offset_pos = gui.add_vector3("TCP offset (m)", initial_value=self.ik.offset.pos, step=0.001)
            self.offset_rot = gui.add_vector3(
                "TCP rotation (deg)", initial_value=self._rpy(self.ik.offset.rot), step=1.0
            )
            self.offset_pos.on_update(self._enqueue(lambda value: self.edit_offset("pos", value)))
            self.offset_rot.on_update(self._enqueue(lambda value: self.edit_offset("rot", self._quat(value))))
        with gui.add_folder("Save / load", order=5):
            self.path = gui.add_text("Setup file", initial_value=str(self.out))
            self._button("Save setup", self.save)
            self._button("Reload setup", self.reload)
            gui.add_markdown(
                "For physics validation:\n```bash\n"
                f"uv run python scripts/play.py --task {self.task} --setup {self.out}\n```"
            )
        self.goal_gizmo = self.server.scene.add_transform_controls("/editor/goal", scale=0.16)
        self.tcp_gizmo = self.server.scene.add_transform_controls("/editor/tcp_offset", scale=0.12, visible=False)
        self.actual_tcp = self.server.scene.add_frame("/editor/actual_tcp", axes_length=0.09, axes_radius=0.002)
        self.desired_tcp = self.server.scene.add_frame("/editor/desired_tcp", axes_length=0.07, axes_radius=0.0015)
        self.goal_gizmo.on_update(self._gizmo_callback("goal"))
        self.tcp_gizmo.on_update(self._gizmo_callback("tcp"))

    def _gizmo_callback(self, which):
        def callback(event):
            if event.client is not None:
                value = (np.array(event.target.position), np.array(event.target.wxyz))
                previous = self.gizmo_poses.get(which)
                if previous is not None:
                    same_position = np.allclose(value[0], previous[0], atol=1e-6, rtol=0)
                    same_rotation = (
                        min(np.linalg.norm(value[1] - previous[1]), np.linalg.norm(value[1] + previous[1])) < 1e-6
                    )
                    if same_position and same_rotation:
                        return
                self.pending.put((self.epoch, lambda pose: self.drag(which, pose), value))

        return callback

    def _rpy(self, quat):
        q = torch.tensor([quat], dtype=torch.float32, device=self.env.device)
        return tuple(math.degrees(float(v[0])) for v in math_utils.euler_xyz_from_quat(q))

    def _quat(self, rpy):
        angles = torch.tensor(rpy, dtype=torch.float32, device=self.env.device) * (math.pi / 180)
        return tuple(float(v) for v in numpy(math_utils.quat_from_euler_xyz(*angles)))

    def _update_pose(self, handle, pose):
        values = pose_to_viser(pose)
        handle.position, handle.wxyz = values["position"], values["wxyz"]

    def rebuild(self):
        for cmd in self.commands:
            validate_command(cmd)
        self.segments = build_segments(self.term, self.commands, self.ik.initial_pose_b())
        for marker in self.markers:
            marker.remove()
        self.markers.clear()
        options = []
        for i, segment in enumerate(self.segments):
            cfg = segment.cfg
            kind = (
                ("Open gripper" if cfg.open_gripper else "Close gripper")
                if isinstance(cfg, GripperCommand)
                else COMMAND_LABELS.get(type(cfg).__name__.removesuffix("Cfg"), type(cfg).__name__.removesuffix("Cfg"))
            )
            label = f"{i}: {kind}"
            if getattr(cfg, "target_frame_name", ""):
                label += f" → {cfg.target_frame_name}"
            options.append(label)
            positions = np.stack([numpy(self.ik.to_world(*segment.sample(f))[0])[0] for f in np.linspace(0, 1, 33)])
            color = (255, 190, 50) if i == self.selected else (100, 170, 240)
            self.markers.append(
                self.server.scene.add_line_segments(
                    f"/editor/path/{i}",
                    points=np.stack((positions[:-1], positions[1:]), axis=1),
                    colors=color,
                    line_width=3.0,
                )
            )
            marker = self.server.scene.add_icosphere(
                f"/editor/bead/{i}",
                radius=0.012 if i != self.selected else 0.019,
                color=color,
                position=positions[-1],
            )
            marker.on_click(self._enqueue(lambda _, index=i: self.select(index)))
            self.markers.append(marker)
        self.selection.options = options
        self.selection.value = options[self.selected]
        if hasattr(self, "add_kind"):
            kinds = self._insert_kinds()
            self.add_kind.options = kinds
            if self.add_kind.value not in kinds:
                self.add_kind.value = kinds[0]

    def select(self, index):
        self.playing = False
        self.selected = max(0, min(index, len(self.commands) - 1))
        self.fraction = 1.0
        self.rebuild()
        self.show_selected()

    def _field(self, name, label, *, boolean=False, vector=False, rotation=False, minimum=None):
        cfg = self.commands[self.selected]
        if not hasattr(cfg, name):
            return
        value = getattr(cfg, name)
        gui = self.server.gui
        if rotation:
            value = self._rpy(value)
        if boolean:
            handle = gui.add_checkbox(label, initial_value=value)
        elif vector or rotation:
            handle = gui.add_vector3(label, initial_value=value, step=1.0 if rotation else 0.001)
        else:
            handle = gui.add_number(label, initial_value=float(value), step=0.01, min=minimum)
        self.command_fields[name] = (handle, rotation)
        handle.on_update(
            self._enqueue(
                lambda value: self.edit(name, self._quat(value) if rotation else value),
                epoch=self.epoch,
            )
        )

    def show_selected(self):
        self.epoch += 1
        self.command_fields = {}
        self.angle_note = None
        self.reference_note = None
        if self.panel is not None:
            self.panel.remove()
        self.panel = self.server.gui.add_folder("Selected command", order=3)
        cfg = self.commands[self.selected]
        with self.panel:
            if isinstance(cfg, GripperCommand):
                self._field("open_gripper", "Open gripper", boolean=True)
                self._field("duration_s", "Hold (s)", minimum=0.01)
            else:
                if isinstance(cfg, (GoToFrameCfg, RotateFrameCfg)) and not isinstance(cfg, ScrewFrameCfg):
                    planned = is_curobo_command(cfg)
                    handle = self.server.gui.add_checkbox("cuRobo plan", initial_value=planned)
                    handle.on_update(self._enqueue(self.set_curobo, epoch=self.epoch))
                    if planned:
                        self.server.gui.add_markdown(
                            "play.py follows cuRobo joint waypoints. This editor still previews the Cartesian path."
                        )
                self._reference_ui(cfg)
                self._field("gripper_open", "Keep gripper open", boolean=True)
                self._field("angle_deg", "Angle (deg)")
                self._field("axial_distance", "Screw travel (m)")
                self._field("axis", "Axis in reference frame", vector=True)
                self._field("use_valve_angle", "Use remaining valve angle", boolean=True)
                with self.server.gui.add_folder("Motion settings", expand_by_default=False):
                    for name, label, minimum in (
                        ("velocity", "Linear speed (m/s)", 0.001),
                        ("angular_velocity", "Angular speed (rad/s)", 0.001),
                        ("distance_threshold", "Position tolerance (m)", 0.0),
                        ("orientation_threshold_deg", "Orientation tolerance (deg)", 0.0),
                        ("angle_threshold_deg", "Arc tolerance (deg)", 0.0),
                    ):
                        self._field(name, label, minimum=minimum)
                    self._field("canonicalize_upward", "Keep TCP upright", boolean=True)
                    self._field("hold_current_orientation", "Hold starting orientation", boolean=True)
                if isinstance(cfg, RotateFrameCfg):
                    angle = float(self.segments[self.selected].handler.angle_rad_tensor[0])
                    self.angle_note = self.server.gui.add_markdown(f"Effective arc: **{math.degrees(angle):.1f}°**")
                if getattr(cfg, "reference_pos_env", None) is not None:
                    self.reference_note = self.server.gui.add_markdown(
                        "Dense cuRobo reference is active. Editing this goal clears that reference."
                    )
                for name in ("position_override_b", "axis_position_override_b"):
                    if getattr(cfg, name, None) is not None:
                        self.server.gui.add_markdown(f"Base-frame constraint `{name}`: `{getattr(cfg, name)}`")
                        self._button(f"Clear {name}", lambda field=name: self.edit(field, None))
        self.scrub.value = self.fraction
        self.sync_gizmos()
        self.preview()

    def _reference_ui(self, cfg):
        fixed = isinstance(cfg, GoToFrameCfg) and cfg.target_position_env is not None
        options = list(self.reference_options)
        if isinstance(cfg, GoToFrameCfg):
            options.insert(0, "Environment (fixed)")
        current = "Environment (fixed)" if fixed else f"{cfg.frame_name}/{cfg.target_frame_name}"
        reference = self.server.gui.add_dropdown("Reference", options=options, initial_value=current)
        reference.on_update(self._enqueue(self.change_reference, epoch=self.epoch))
        if fixed:
            self._field("target_position_env", "Goal in environment (m)", vector=True)
            if cfg.target_orientation_env is None:
                self.server.gui.add_markdown(
                    "Orientation holds its starting value. Dragging authors an explicit rotation."
                )
            else:
                self._field("target_orientation_env", "Goal rotation (deg)", rotation=True)
        else:
            self._field("target_offset_pos", "Offset in reference (m)", vector=True)
            self._field("target_offset_rot", "Rotation in reference (deg)", rotation=True)

    def sync_gizmos(self):
        cfg = self.commands[self.selected]
        segment = self.segments[self.selected]
        self.goal_gizmo.visible = not isinstance(cfg, GripperCommand) and not self.calibrate.value
        self.tcp_gizmo.visible = self.calibrate.value
        pose = segment.pivot if isinstance(cfg, RotateFrameCfg) else segment.end
        self._update_pose(self.goal_gizmo, self.ik.to_world(*pose))
        self._update_pose(self.tcp_gizmo, self.ik.tcp_pose_w())
        for name, handle in (("goal", self.goal_gizmo), ("tcp", self.tcp_gizmo)):
            self.gizmo_poses[name] = (np.array(handle.position), np.array(handle.wxyz))
        if isinstance(cfg, RotateFrameCfg):
            center = numpy(self.ik.to_world(*segment.pivot)[0])[0]
            axis = numpy(
                math_utils.quat_apply(
                    tensor(self.ik.robot.data.root_quat_w),
                    segment.handler.rot_axis_b,
                )
            )[0]
            points = np.array([[center - axis * 0.15, center + axis * 0.15]])
        else:
            points = np.zeros((1, 2, 3))
        self.server.scene.add_line_segments(
            "/editor/rotation_axis",
            points=points,
            colors=(210, 80, 230),
            line_width=4.0,
            visible=isinstance(cfg, RotateFrameCfg),
        )

    def preview(self):
        segment = self.segments[self.selected]
        goal = segment.sample(self.fraction)
        if not self.calibrate.value:
            start = time.perf_counter()
            pos_error, rot_error = self.ik.solve(goal, segment.gripper_open, self.ik_iters)
            elapsed = (time.perf_counter() - start) * 1000
            result = "Reached" if pos_error < 0.005 and rot_error < 3 else "IK residual — inspect reach / joint limits"
            self.status.content = (
                f"{'**Unsaved** · ' if self.dirty else ''}{result}\n\n"
                f"Position **{pos_error * 1000:.1f} mm** · Rotation **{rot_error:.1f}°** · Solve {elapsed:.0f} ms"
            )
        else:
            self.status.content = "**TCP calibration** — the robot stays still while you move the offset."
        self._update_pose(self.actual_tcp, self.ik.tcp_pose_w())
        self._update_pose(self.desired_tcp, self.ik.to_world(*goal))
        if self.angle_note is not None:
            angle = float(segment.handler.angle_rad_tensor[0])
            self.angle_note.content = f"Effective arc: **{math.degrees(angle):.1f}°**"
        if self.reference_note is not None:
            self.reference_note.visible = getattr(segment.cfg, "reference_pos_env", None) is not None
        self.sync_gizmos()

    def _clear_reference_path(self, cfg):
        if getattr(cfg, "reference_pos_env", None) is not None:
            cfg.reference_pos_env = None
            cfg.reference_quat_xyzw = None

    def edit(self, name, value):
        candidate = copy.deepcopy(self.commands[self.selected])
        setattr(candidate, name, value)
        if name in {"target_offset_pos", "target_offset_rot", "target_position_env", "target_orientation_env"}:
            self._clear_reference_path(candidate)
        validate_command(candidate)
        self.commands[self.selected] = candidate
        self.changed(refresh_panel=name in {"position_override_b", "axis_position_override_b"})

    def changed(self, *, refresh_panel=True):
        self.dirty, self.playing = True, False
        self.rebuild()
        if refresh_panel:
            self.show_selected()
        else:
            # Preserve the input handles and keyboard focus while a value is typed.
            self.preview()

    def change_reference(self, choice):
        cfg = self.commands[self.selected]
        pose = (
            self.segments[self.selected].pivot if isinstance(cfg, RotateFrameCfg) else self.segments[self.selected].end
        )
        pose_w = self.ik.to_world(*pose)
        if choice == "Environment (fixed)":
            cfg.frame_name = cfg.target_frame_name = ""
            cfg.target_position_env = tuple(float(v) for v in numpy(pose_w[0] - self.env.scene.env_origins)[0])
            cfg.target_orientation_env = tuple(float(v) for v in numpy(pose_w[1])[0])
        else:
            cfg.frame_name, cfg.target_frame_name = self.reference_options[choice]
            if isinstance(cfg, GoToFrameCfg):
                cfg.target_position_env = cfg.target_orientation_env = None
            self._store_relative_pose(cfg, pose_w)
        self._clear_reference_path(cfg)
        self.changed()

    def _store_relative_pose(self, cfg, pose_w):
        sensor = self.env.scene[cfg.frame_name]
        index = sensor.data.target_frame_names.index(cfg.target_frame_name)
        offset = math_utils.subtract_frame_transforms(
            tensor(sensor.data.target_pos_w)[:, index],
            tensor(sensor.data.target_quat_w)[:, index],
            *pose_w,
        )
        cfg.target_offset_pos = tuple(float(v) for v in numpy(offset[0])[0])
        cfg.target_offset_rot = tuple(float(v) for v in numpy(offset[1])[0])

    def drag(self, which, value):
        self.playing = False
        added_orientation = False
        pose_w = viser_to_pose(*value, device=self.env.device)
        if which == "tcp":
            offset = math_utils.subtract_frame_transforms(
                tensor(self.ik.robot.data.body_pos_w)[:, self.ik.body_idx],
                tensor(self.ik.robot.data.body_quat_w)[:, self.ik.body_idx],
                *pose_w,
            )
            self.ik.offset.pos = tuple(float(v) for v in numpy(offset[0])[0])
            self.ik.offset.rot = tuple(float(v) for v in numpy(offset[1])[0])
            self.offset_pos.value = self.ik.offset.pos
            self.offset_rot.value = self._rpy(self.ik.offset.rot)
        else:
            cfg = self.commands[self.selected]
            if isinstance(cfg, GoToFrameCfg) and cfg.target_position_env is not None:
                added_orientation = cfg.target_orientation_env is None
                cfg.target_position_env = tuple(float(v) for v in numpy(pose_w[0] - self.env.scene.env_origins)[0])
                cfg.target_orientation_env = tuple(float(v) for v in numpy(pose_w[1])[0])
            else:
                self._store_relative_pose(cfg, pose_w)
            self._clear_reference_path(cfg)
        # Keep the same UI generation while a drag is in flight.
        self.dirty = True
        self.rebuild()
        for name, (handle, rotation) in self.command_fields.items():
            value = getattr(self.commands[self.selected], name)
            handle.value = self._rpy(value) if rotation else value
        if added_orientation:
            self.show_selected()
        else:
            self.preview()

    def edit_offset(self, name, value):
        data = self.payload()
        data["body_offset"][name] = list(value)
        _, offset = validate_setup(data)
        self.ik.offset = SequentialPoseCommandCfg.OffsetCfg(**offset)
        self.changed(refresh_panel=False)

    def set_calibrating(self, value):
        self.playing = False
        self.sync_gizmos()
        self.preview()

    def seek(self, value):
        self.playing, self.fraction = False, float(value)
        self.preview()

    def play(self):
        self.calibrate.value = False
        self.ik.reset()
        self.selected, self.fraction = 0, 0.0
        self.rebuild()
        self.show_selected()
        self.playing = True

    def reset(self):
        self.playing = False
        self.ik.reset()
        self.status.content = "Robot restored to its reset joint pose. Select a command to preview."
        self._update_pose(self.actual_tcp, self.ik.tcp_pose_w())
        self.sync_gizmos()

    def duplicate(self):
        self.commands.insert(self.selected + 1, copy.deepcopy(self.commands[self.selected]))
        self.selected += 1
        self.changed()

    def move(self, delta):
        target = self.selected + delta
        if 0 <= target < len(self.commands):
            self.commands[self.selected], self.commands[target] = self.commands[target], self.commands[self.selected]
            self.selected = target
            self.changed()

    def delete(self):
        if len(self.commands) == 1:
            raise ValueError("Keep at least one command")
        self.commands.pop(self.selected)
        self.selected = min(self.selected, len(self.commands) - 1)
        self.changed()

    def _insert_kinds(self) -> list[str]:
        kinds = ["GoTo", "Curobo GoTo", "Open gripper", "Close gripper", "Rotate", "Curobo Rotate"]
        if any(is_curobo_command(cmd) for cmd in self.commands):
            return ["Curobo GoTo", "Curobo Rotate", "Open gripper", "Close gripper", "GoTo", "Rotate"]
        return kinds

    def _planner_settings(self) -> dict:
        return planner_settings((*self.commands, *self.term.cfg.commands))

    def set_curobo(self, enabled: bool):
        cfg = self.commands[self.selected]
        self.commands[self.selected] = (
            as_curobo_command(cfg, self._planner_settings()) if enabled else as_direct_command(cfg)
        )
        self.changed()

    def insert(self):
        kind = self.add_kind.value
        if "gripper" in kind:
            cfg = GripperCommand(open_gripper=kind.startswith("Open"), duration_s=0.3)
        elif "Rotate" in kind:
            if not self.reference_options:
                raise ValueError("Rotate needs a scene frame sensor")
            frame, target = next(iter(self.reference_options.values()))
            cfg = RotateFrameCfg(frame_name=frame, target_frame_name=target, use_valve_angle=False)
            if kind.startswith("Curobo"):
                cfg = as_curobo_command(cfg, self._planner_settings())
        else:
            pose = self.ik.tcp_pose_w()
            cfg = GoToFrameCfg(
                frame_name="",
                target_frame_name="",
                canonicalize_upward=False,
                target_position_env=tuple(float(v) for v in numpy(pose[0] - self.env.scene.env_origins)[0]),
                target_orientation_env=tuple(float(v) for v in numpy(pose[1])[0]),
            )
            if kind.startswith("Curobo"):
                cfg = as_curobo_command(cfg, self._planner_settings())
        self.commands.insert(self.selected + 1, cfg)
        self.selected += 1
        self.changed()

    def payload(self):
        return make_setup(self.task, self.term.cfg, self.commands, self.ik.offset)

    def save(self):
        self.out = Path(self.path.value).expanduser()
        save_setup(self.out, self.payload())
        self.dirty = False
        self.status.content = f"Saved **{self.out}**. Load with `play.py --task {self.task} --setup {self.out}`."

    def reload(self):
        data = load_setup(Path(self.path.value).expanduser())
        # Validate task, robot and frame references against a disposable config.
        apply_setup(copy.deepcopy(self.env.cfg), data, task=self.task)
        self.commands, offset = validate_setup(data, task=self.task)
        self.ik.offset = SequentialPoseCommandCfg.OffsetCfg(**offset)
        self.offset_pos.value, self.offset_rot.value = self.ik.offset.pos, self._rpy(self.ik.offset.rot)
        self.selected, self.dirty, self.playing = 0, False, False
        self.fraction = 1.0
        self.rebuild()
        self.show_selected()

    def poll(self, dt):
        # Coalesce queued drag updates: one IK solve per latest drag, not per mouse event.
        drag = None
        while not self.pending.empty():
            epoch, callback, value = self.pending.get_nowait()
            if epoch is not None and epoch != self.epoch:
                continue
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], np.ndarray):
                drag = (epoch, callback, value)
            else:
                callback(value)
        if drag is not None and drag[0] == self.epoch:
            drag[1](drag[2])
        if self.playing:
            self.fraction += dt / self.segments[self.selected].duration
            if self.fraction >= 1:
                self.fraction = 1.0
                self.preview()
                if self.selected == len(self.commands) - 1:
                    self.playing = False
                else:
                    self.selected += 1
                    self.fraction = 0.0
                    self.rebuild()
                    self.show_selected()
            else:
                self.preview()
            self.scrub.value = self.fraction


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--setup", help="Load an existing command setup JSON.")
    parser.add_argument("--out", help="Save path (default: --setup or logs/command_setup.json).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ik-iters", type=int, default=24)
    parser.add_argument("--smoke-test", action="store_true", help="Construct scene/UI, solve once and exit.")
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    # A single kinematic preview is small; CPU avoids CUDA solver startup and
    # works on machines used to author tasks away from the training workstation.
    args.device = args.device or "cpu"
    if args.ik_iters < 1:
        parser.error("--ik-iters must be positive")
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    args.visualizer = None
    args.visualizer_explicit = args.visualizer_disable_all = True
    return args, hydra_args


def main():
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    if not isinstance(getattr(env_cfg.commands, "pose_command", None), SequentialPoseCommandCfg):
        raise SystemExit("Choose a task with SequentialPoseCommandCfg (GoTo / gripper / Rotate).")
    if args.setup:
        apply_setup(env_cfg, load_setup(args.setup), task=args.task)
    env_cfg.scene.num_envs = 1
    env_cfg.recorders = None
    env_cfg.commands.pose_command.class_type = _PreviewCommand
    env_cfg.commands.pose_command.debug_vis = False
    env_cfg.commands.pose_command.path_debug_vis = False
    env_cfg.sim.visualizer_cfgs = None
    if args.device == "cpu":
        # Only FK / Jacobians are used: avoid creating the CUDA-only MJWarp solver.
        from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

        env_cfg.sim.physics = NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), use_cuda_graph=False)
        env_cfg.sim.device = "cpu"
    out = Path(args.out or args.setup or "logs/command_setup.json")
    from isaaclab_newton.physics import NewtonManager
    from newton.viewer import ViewerViser

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        viewer = None
        try:
            env.reset()
            refresh_frame_sensors(env.unwrapped)
            viewer = ViewerViser(port=args.port, label="HiveBoard command setup")
            viewer.set_model(NewtonManager.get_model())
            editor = CommandEditor(env.unwrapped, viewer, args.task, out, args.ik_iters)
            center = numpy(editor.ik.initial_flange[0])[0]
            viewer._server.initial_camera.position = center + np.array([1.1, -1.1, 0.7])
            viewer._server.initial_camera.look_at = center
            print(f"[INFO] Command editor: {viewer.url}\n[INFO] Save to {out}", flush=True)
            previous = time.perf_counter()
            while viewer.is_running():
                start = time.perf_counter()
                try:
                    editor.poll(min(start - previous, 0.1))
                except (ValueError, RuntimeError, OSError, KeyError) as exc:
                    editor.playing = False
                    editor.status.content = f"**Cannot apply edit:** {exc}"
                    print(f"[EDITOR] {exc}", flush=True)
                previous = start
                viewer.begin_frame(start)
                viewer.log_state(NewtonManager.get_state())
                viewer.end_frame()
                if args.smoke_test:
                    print(f"[SMOKE] {len(editor.commands)} commands; {editor.status.content}", flush=True)
                    break
                time.sleep(max(0, 1 / 30 - (time.perf_counter() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            if viewer is not None:
                viewer.close()
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
