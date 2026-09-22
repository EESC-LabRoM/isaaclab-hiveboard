#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drag command goals, rotation references and the tool offset in Newton/Viser.

    uv run python scripts/command_edit.py
    uv run python scripts/command_edit.py --setup logs/command_setup.json

This is a kinematic authoring preview. GoTo and Rotate commands always use
cuRobo; the editor plans automatically so the path and robot pose are the
same joint waypoints play.py executes.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import queue
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# Avoid concurrent USD authoring races in the kitless scene importer. Respect an
# explicit user setting. This must precede imports that initialize OpenUSD/TBB.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

import gymnasium as gym
import isaaclab_hiveboard  # noqa: F401
import numpy as np
import torch
from isaaclab_hiveboard.assets import ASSET_DIR, DYNAARM_EE_LINK, FRANKA_EE, SPOT_EE
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    SequentialPoseCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
from isaaclab_hiveboard.utils.command_preview import (
    PreviewIK,
    Segment,
    apply_authored_command_field,
    build_segments,
    effective_arc_caption,
    format_editor_plan_failure,
    format_ik_debug,
    invalidate_preview_plans_from,
    numpy,
    plan_curobo_segments,
    pose_to_viser,
    refresh_frame_sensors,
    tensor,
    viser_to_pose,
)
from isaaclab_hiveboard.utils.command_setup import (
    apply_setup,
    as_curobo_command,
    default_setup_path,
    is_curobo_command,
    load_setup,
    make_setup,
    planner_settings,
    save_setup,
    validate_command,
    validate_setup,
)

import isaaclab.utils.math as math_utils
from isaaclab.sensors import BaseFrameTransformer

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

DEFAULT_TASK = "Isaac-HiveBoard-Spot-BallValve-Play-v0"
COMMAND_LABELS = {
    "CuroboPlannedGoToFrame": "cuRobo GoTo",
    "CuroboPlannedRotateFrame": "cuRobo Rotate",
    "ScrewFrame": "Screw",
}
# Keyed by the body each cuRobo model plans for, which is the body a task's pose
# command drives. On ANYmal that is the DynaArm flange, not the gripper palm
# ANYMAL_EE names.
BUNDLED_ROBOT_MODELS = {
    FRANKA_EE.body_name: "franka/cumotion/fr3.yaml",
    SPOT_EE.body_name: "spot/cumotion/spot_arm.yaml",
    DYNAARM_EE_LINK: "anymal/cumotion/dynaarm.yaml",
}
ENVIRONMENT_REFERENCE = "Environment (fixed)"
INSERT_KINDS = ("Curobo GoTo", "Curobo Rotate", "Open gripper", "Close gripper")
MOTION_FIELDS = (
    ("velocity", "Linear speed (m/s)", 0.001),
    ("angular_velocity", "Angular speed (rad/s)", 0.001),
    ("distance_threshold", "Position tolerance (m)", 0.0),
    ("orientation_threshold_deg", "Orientation tolerance (deg)", 0.0),
    ("angle_threshold_deg", "Arc tolerance (deg)", 0.0),
)
# Editing any of these moves the authored goal away from a retargeted path.
GOAL_FIELDS = frozenset({"target_offset_pos", "target_offset_rot", "target_position_env", "target_orientation_env"})
# Editing any of these changes the geometry cuRobo planned against.
REPLAN_FIELDS = GOAL_FIELDS | {"angle_deg", "max_ee_rotation_deg", "axis", "axial_distance", "use_valve_angle"}
PATH_COLOR = (100, 170, 240)
SELECTED_PATH_COLOR = (255, 190, 50)
ROTATION_AXIS_COLOR = (210, 80, 230)
PATH_SAMPLES = 33
REACHED_POSITION_MM = 5.0
REACHED_ROTATION_DEG = 3.0


def _bundled_planner_settings(body_name: str | None) -> dict:
    """Planner settings read from the cuRobo model shipped with a known end effector."""
    # Only the bundled robots have a model the editor can configure on its own.
    if body_name not in BUNDLED_ROBOT_MODELS:
        raise ValueError(f"No bundled cuRobo robot model for body {body_name!r}.")
    yaml_path = Path(ASSET_DIR) / BUNDLED_ROBOT_MODELS[body_name]
    kinematics = load_curobo_robot_cfg(yaml_path)["robot_cfg"]["kinematics"]
    return {
        "robot_joint_names": list(kinematics["cspace"]["joint_names"]),
        "robot_curobo_yaml": str(yaml_path),
        "robot_urdf": str((yaml_path.parent / kinematics["urdf_path"]).resolve()),
    }


def _planner_settings(commands, body_name=None) -> dict:
    """Planner settings of the sequence's own cuRobo commands, else the bundled model.

    A task may plan for a body the bundled table does not name, and its authored
    commands already carry the model that planned them, so they come first.
    """
    try:
        return planner_settings(commands)
    except ValueError:
        return _bundled_planner_settings(body_name)


def _needs_curobo(cfg) -> bool:
    """True for a plain GoTo / Rotate the editor has to upgrade before previewing."""
    return (
        isinstance(cfg, (GoToFrameCfg, RotateFrameCfg))
        and not isinstance(cfg, ScrewFrameCfg)
        and not is_curobo_command(cfg)
    )


def _curobo_commands(commands, *, body_name=None):
    """Upgrade plain GoTo/Rotate commands using the known robot's bundled model."""
    commands = copy.deepcopy(list(commands))
    # Resolving the settings loads a robot model, so skip it for ready sequences.
    if not any(_needs_curobo(cfg) for cfg in commands):
        return commands
    planner = _planner_settings(commands, body_name)
    return [as_curobo_command(cfg, planner) if _needs_curobo(cfg) else cfg for cfg in commands]


class _PreviewCommand(SequentialPoseCommand):
    """Keep reset from launching cuRobo plans before the editor opens."""

    def _resample_command(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        self._sample_valve_task(ids)
        pos, quat = self._get_ee_in_base_frame(ids)
        self._command[:, 1:4] = pos
        self._command[:, 4:8] = quat


def _values(vector) -> tuple[float, ...]:
    """First environment's row of a tensor, as the plain floats a command stores."""
    return tuple(float(v) for v in numpy(vector)[0])


def _preview_plan(cfg) -> dict | None:
    """Cached cuRobo waypoints; the planner attaches them to a command config."""
    return getattr(cfg, "_preview_plan", None)


def _reference_path(cfg):
    """Dense authored TCP path, only carried by retargeted cuRobo GoTo commands."""
    # Rotations and plain goals have no dense reference to follow.
    if not isinstance(cfg, CuroboPlannedGoToFrameCfg):
        return None
    return cfg.reference_pos_env


def _axis_position_override(cfg):
    """Base-frame pivot constraint, only carried by rotation commands."""
    # Goals have no rotation axis to constrain.
    if not isinstance(cfg, RotateFrameCfg):
        return None
    return cfg.axis_position_override_b


def _command_label(index: int, cfg) -> str:
    """Sequence dropdown entry, such as ``2: cuRobo GoTo → valve_handle``."""
    # A gripper command has no goal frame and reads better as its action.
    if isinstance(cfg, GripperCommand):
        return f"{index}: {'Open gripper' if cfg.open_gripper else 'Close gripper'}"
    kind = type(cfg).__name__.removesuffix("Cfg")
    label = f"{index}: {COMMAND_LABELS.get(kind, kind)}"
    # A goal fixed to the environment origin names no reference frame.
    if cfg.target_frame_name:
        label += f" → {cfg.target_frame_name}"
    return label


def _frame_reference_options(sensors) -> dict[str, tuple[str, str]]:
    """Selectable ``sensor/frame`` anchors, keyed by the label shown in the UI."""
    options = {}
    for sensor_name, sensor in sensors.items():
        # Only frame transformers publish poses, and ee_frame tracks the tool itself.
        if not isinstance(sensor, BaseFrameTransformer) or sensor_name == "ee_frame":
            continue
        for frame in sensor.cfg.target_frames:
            options[f"{sensor_name}/{frame.name}"] = (sensor_name, frame.name)
    return options


def _same_pose(pose, previous) -> bool:
    """Compare two Viser poses, reading ``q`` and ``-q`` as the same rotation."""
    # The first event of a gizmo has nothing to compare against.
    if previous is None:
        return False
    position, wxyz = pose
    previous_position, previous_wxyz = previous
    same_position = np.allclose(position, previous_position, atol=1e-6, rtol=0)
    same_rotation = min(np.linalg.norm(wxyz - previous_wxyz), np.linalg.norm(wxyz + previous_wxyz)) < 1e-6
    return same_position and same_rotation


@dataclass(frozen=True)
class _QueuedEdit:
    """One captured interaction waiting to run on the simulation thread."""

    run: Callable[[], None]
    generation: int | None
    is_drag: bool = False


class GuiEvents:
    """Capture Viser interactions and replay them on the simulation thread.

    Viser calls its handlers from a network thread while the editor writes
    robot state and solves IK. Widget values are copied inside the handler and
    the work runs later, from :meth:`drain`.
    """

    def __init__(self):
        self._queue: queue.SimpleQueue[_QueuedEdit] = queue.SimpleQueue()
        self._drag_poses: dict[str, tuple] = {}
        self.generation = 0

    def new_generation(self) -> None:
        """Retire the handlers of widgets the editor is about to replace."""
        self.generation += 1

    def submit(self, action: Callable[[], None]) -> None:
        """Queue editor-side work for the next :meth:`drain`."""
        self._queue.put(_QueuedEdit(action, None))

    def on_click(self, action: Callable[[], None]):
        """Viser click handler that runs ``action`` without a widget value."""

        def handler(event):
            # Programmatic updates carry no client and must not re-enter the editor.
            if event.client is None:
                return
            self._queue.put(_QueuedEdit(action, None))

        return handler

    def on_change(self, action: Callable[[object], None], *, versioned: bool = False):
        """Viser update handler forwarding the new widget value to ``action``.

        A versioned handler is dropped once the panel it belongs to is rebuilt.
        """
        generation = self.generation if versioned else None

        def handler(event):
            # Programmatic updates carry no client and must not re-enter the editor.
            if event.client is None:
                return
            value = copy.deepcopy(event.target.value)
            self._queue.put(_QueuedEdit(partial(action, value), generation))

        return handler

    def on_drag(self, name: str, action: Callable[[tuple, str], None]):
        """Viser gizmo handler forwarding the dragged ``(position, wxyz)`` pose and phase.

        ``action`` receives the pose and the gizmo's drag phase (``"start"``,
        ``"update"`` or ``"end"``), so it can defer expensive work until release.
        """

        def handler(event):
            # Programmatic updates carry no client and must not re-enter the editor.
            if event.client is None:
                return
            pose = (np.array(event.target.position), np.array(event.target.wxyz))
            # Viser echoes back the pose the editor just wrote; that is not a drag.
            # Release always matters, even if it lands on the same pose as the last update.
            if event.phase != "end" and _same_pose(pose, self._drag_poses.get(name)):
                return
            self._queue.put(_QueuedEdit(partial(action, pose, event.phase), self.generation, is_drag=True))

        return handler

    def remember_drag_pose(self, name: str, handle) -> None:
        """Record a gizmo pose the editor wrote, so its echo is ignored."""
        self._drag_poses[name] = (np.array(handle.position), np.array(handle.wxyz))

    def drain(self) -> None:
        """Run the queued edits, keeping only the newest pose of a drag."""
        latest_drag = None
        while not self._queue.empty():
            edit = self._queue.get_nowait()
            # A rebuilt panel invalidates the values typed into its old widgets.
            if edit.generation is not None and edit.generation != self.generation:
                continue
            if edit.is_drag:
                latest_drag = edit
            else:
                edit.run()
        # Coalescing drags costs one IK solve per frame instead of one per mouse event.
        if latest_drag is not None and latest_drag.generation == self.generation:
            latest_drag.run()


class SequencePaths:
    """Path polyline and endpoint bead of every command, drawn in the Viser scene."""

    def __init__(self, scene, events: GuiEvents, select: Callable[[int], None]):
        self._scene = scene
        self._events = events
        self._select = select
        self._handles = []

    def redraw(self, paths: list[np.ndarray], selected: int) -> None:
        """Replace all paths by world positions, highlighting the selected command."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for index, positions in enumerate(paths):
            highlighted = index == selected
            self._handles.append(self._draw_line(index, positions, highlighted))
            self._handles.append(self._draw_bead(index, positions[-1], highlighted))

    def _draw_line(self, index: int, positions: np.ndarray, highlighted: bool):
        return self._scene.add_line_segments(
            f"/editor/path/{index}",
            points=np.stack((positions[:-1], positions[1:]), axis=1),
            colors=SELECTED_PATH_COLOR if highlighted else PATH_COLOR,
            line_width=3.0,
        )

    def _draw_bead(self, index: int, position: np.ndarray, highlighted: bool):
        bead = self._scene.add_icosphere(
            f"/editor/bead/{index}",
            radius=0.019 if highlighted else 0.012,
            color=SELECTED_PATH_COLOR if highlighted else PATH_COLOR,
            position=position,
        )
        bead.on_click(self._events.on_click(partial(self._select, index)))
        return bead


class CommandEditor:
    """Authoring UI for one task's command sequence and its tool offset."""

    def __init__(self, env, viewer, task: str, out: Path, ik_iters: int):
        self.env, self.task, self.out = env, task, out
        # Newton currently exposes its Viser server through this single private attribute.
        self.server = viewer._server
        self.term = env.command_manager.get_term("pose_command")
        self.commands = copy.deepcopy(list(self.term.cfg.commands))
        self.ik = PreviewIK(env, self.term)
        # A task without an authored tool offset starts calibration from identity.
        self.ik.offset = self.ik.offset or SequentialPoseCommandCfg.OffsetCfg()
        self.ik_iters = ik_iters
        self.selected, self.fraction = 0, 1.0
        self.playing, self.dirty = False, False
        self.events = GuiEvents()
        self.panel = None
        self.command_fields = {}
        self.angle_note = None
        self.reference_note = None
        self.reference_options = _frame_reference_options(env.scene.sensors)
        self._create_ui()
        self.rebuild()
        self._start_preview()

    # ------------------------------------------------------------------ setup

    def _start_preview(self) -> None:
        """Show the first command, planning first when the sequence needs cuRobo."""
        # A sequence without cuRobo commands can be previewed straight away.
        if not any(is_curobo_command(cmd) for cmd in self.commands):
            self.show_selected()
            return
        self.refresh_selected_panel()
        self._update_pose(self.actual_tcp, self.ik.tcp_pose_w())
        self._update_pose(self.desired_tcp, self.ik.to_world(*self.segments[self.selected].end))
        self.status.content = "Robot at reset pose. Preparing cuRobo preview…"
        # main() sends the reset pose and camera before poll() runs this.
        self.events.submit(self.plan_with_curobo)

    def _create_ui(self) -> None:
        self._create_header()
        self._create_sequence_folder()
        self._create_tcp_folder()
        self._create_setup_folder()
        self._create_scene_handles()

    def _create_header(self) -> None:
        gui = self.server.gui
        gui.add_markdown(
            "## HiveBoard command setup\n"
            "Drag a goal or rotation reference. RGB axes are X/Y/Z. "
            "**Kinematic preview**: objects stay at their reset poses. "
            "GoTo and Rotate use cuRobo; the editor plans automatically to match the joint path play.py runs.",
            order=0,
        )
        self.status = gui.add_markdown("Loading…", order=1)

    def _create_sequence_folder(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Sequence", order=2):
            self.selection = gui.add_dropdown("Command", options=["Loading"])
            self.selection.on_update(self.events.on_change(self._select_labelled))
            self.scrub = gui.add_slider("Progress", min=0.0, max=1.0, step=0.01, initial_value=1.0)
            self.scrub.on_update(self.events.on_change(self.seek))
            self._button("Preview sequence", self.play)
            self._button("Pause", self.pause)
            self._button("Reset robot pose", self.reset)
            self._button("Dump IK debug", self.dump_ik_debug)
            self._create_edit_folder()

    def _create_edit_folder(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Edit sequence", expand_by_default=False):
            self._button("Duplicate command", self.duplicate)
            self._button("Move earlier", partial(self.move, -1))
            self._button("Move later", partial(self.move, 1))
            self._button("Delete command", self.delete)
            self.add_kind = gui.add_dropdown("New command", options=list(INSERT_KINDS))
            self._button("Insert after selected", self.insert)

    def _create_tcp_folder(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Tool center point (TCP)", order=4, expand_by_default=False):
            gui.add_markdown(
                f"Offset relative to **{self.term.cfg.body_name}**. Rotations below are XYZ Euler degrees."
            )
            self.calibrate = gui.add_checkbox("Drag TCP offset", initial_value=False)
            self.calibrate.on_update(self.events.on_change(self.set_calibrating))
            self.offset_pos = gui.add_vector3("TCP offset (m)", initial_value=self.ik.offset.pos, step=0.001)
            self.offset_rot = gui.add_vector3(
                "TCP rotation (deg)", initial_value=self._euler_degrees(self.ik.offset.rot), step=1.0
            )
            self.offset_pos.on_update(self.events.on_change(partial(self.edit_offset, "pos")))
            self.offset_rot.on_update(self.events.on_change(self._edit_offset_rotation))

    def _create_setup_folder(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Save / load", order=5):
            self.path = gui.add_text("Setup file", initial_value=str(self.out))
            self._button("Save setup", self.save)
            self._button("Reload setup", self.reload)
            gui.add_markdown(
                "For physics validation:\n```bash\n"
                f"uv run python scripts/play.py --task {self.task} --setup {self.out}\n```"
            )

    def _create_scene_handles(self) -> None:
        scene = self.server.scene
        self.goal_gizmo = scene.add_transform_controls("/editor/goal", scale=0.16)
        self.tcp_gizmo = scene.add_transform_controls("/editor/tcp_offset", scale=0.12, visible=False)
        self.actual_tcp = scene.add_frame("/editor/actual_tcp", axes_length=0.09, axes_radius=0.002)
        self.desired_tcp = scene.add_frame("/editor/desired_tcp", axes_length=0.07, axes_radius=0.0015)
        self.goal_gizmo.on_update(self.events.on_drag("goal", self.drag_goal))
        self.tcp_gizmo.on_update(self.events.on_drag("tcp", self.drag_tcp_offset))
        self.paths = SequencePaths(scene, self.events, self.select)

    def _button(self, label: str, action: Callable[[], None]) -> None:
        self.server.gui.add_button(label).on_click(self.events.on_click(action))

    # ------------------------------------------------------- unit conversions

    def _euler_degrees(self, quat) -> tuple[float, ...]:
        """XYZ Euler degrees of an ``xyzw`` quaternion, as the GUI shows rotations."""
        q = torch.tensor([quat], dtype=torch.float32, device=self.env.device)
        return tuple(math.degrees(float(v[0])) for v in math_utils.euler_xyz_from_quat(q))

    def _quat_from_degrees(self, rpy) -> tuple[float, ...]:
        """``xyzw`` quaternion of XYZ Euler degrees typed into the GUI."""
        angles = torch.tensor(rpy, dtype=torch.float32, device=self.env.device) * (math.pi / 180)
        return tuple(float(v) for v in numpy(math_utils.quat_from_euler_xyz(*angles)))

    def _update_pose(self, handle, pose) -> None:
        values = pose_to_viser(pose)
        handle.position, handle.wxyz = values["position"], values["wxyz"]

    # -------------------------------------------------------------- sequence

    def rebuild(self) -> None:
        """Re-resolve the sequence geometry and redraw it in the scene."""
        for cmd in self.commands:
            validate_command(cmd)
        self.segments = build_segments(self.term, self.commands, self.ik.initial_pose_b())
        self.paths.redraw([self._path_of(segment) for segment in self.segments], self.selected)
        options = [_command_label(index, segment.cfg) for index, segment in enumerate(self.segments)]
        self.selection.options = options
        self.selection.value = options[self.selected]

    def _path_of(self, segment: Segment) -> np.ndarray:
        """World positions sampled along one command, for its scene polyline."""
        fractions = np.linspace(0, 1, PATH_SAMPLES)
        return np.stack([numpy(self.ik.to_world(*segment.sample(f))[0])[0] for f in fractions])

    def _select_labelled(self, label: str) -> None:
        self.select(int(label.split(":", 1)[0]))

    def select(self, index: int) -> None:
        """Make ``index`` the edited command and preview its end pose."""
        self.playing = False
        self.selected = max(0, min(index, len(self.commands) - 1))
        self.fraction = 1.0
        self.rebuild()
        self.show_selected()

    def duplicate(self) -> None:
        self.commands.insert(self.selected + 1, copy.deepcopy(self.commands[self.selected]))
        self.selected += 1
        self.changed(replan=is_curobo_command(self.commands[self.selected]))

    def move(self, delta: int) -> None:
        """Swap the selected command with its neighbour ``delta`` steps away."""
        target = self.selected + delta
        # The ends of the sequence have no neighbour to swap with.
        if 0 <= target < len(self.commands):
            swapped = (self.commands[self.selected], self.commands[target])
            self.commands[self.selected], self.commands[target] = self.commands[target], self.commands[self.selected]
            self.selected = target
            # Reordering motions re-chains every plan from the swap onward.
            self.changed(replan=not all(isinstance(cmd, GripperCommand) for cmd in swapped))

    def delete(self) -> None:
        # An empty sequence has no geometry to preview.
        if len(self.commands) == 1:
            raise ValueError("Keep at least one command")
        removed = self.commands.pop(self.selected)
        self.selected = min(self.selected, len(self.commands) - 1)
        # A gripper hold keeps the TCP still, so the plans around it still chain.
        self.changed(replan=not isinstance(removed, GripperCommand))

    def insert(self) -> None:
        """Add the command chosen in the ``New command`` dropdown after the selected one."""
        cfg = self._new_command(self.add_kind.value)
        self.commands.insert(self.selected + 1, cfg)
        self.selected += 1
        self.changed(replan=is_curobo_command(cfg))

    def _new_command(self, kind: str):
        # Gripper holds are the only commands without a planned motion.
        if "gripper" in kind:
            return GripperCommand(open_gripper=kind.startswith("Open"), duration_s=0.3)
        cfg = self._new_rotate() if "Rotate" in kind else self._new_goto()
        return as_curobo_command(cfg, _planner_settings(self.commands, self.term.cfg.body_name))

    def _new_rotate(self) -> RotateFrameCfg:
        """Rotation around the first reference frame the scene offers."""
        # Without a frame sensor there is no pivot to rotate around.
        if not self.reference_options:
            raise ValueError("Rotate needs a scene frame sensor")
        frame, target = next(iter(self.reference_options.values()))
        return RotateFrameCfg(frame_name=frame, target_frame_name=target, use_valve_angle=False)

    def _new_goto(self) -> GoToFrameCfg:
        """Goal pinned to the environment at the robot's current TCP pose."""
        pose = self.ik.tcp_pose_w()
        return GoToFrameCfg(
            frame_name="",
            target_frame_name="",
            canonicalize_upward=False,
            target_position_env=_values(pose[0] - self.env.scene.env_origins),
            target_orientation_env=_values(pose[1]),
        )

    # ---------------------------------------------------------- command panel

    def show_selected(self) -> None:
        self.refresh_selected_panel()
        self.preview()

    def refresh_selected_panel(self) -> None:
        """Rebuild the widgets bound to the selected command."""
        self.events.new_generation()
        self.command_fields = {}
        self.angle_note = None
        self.reference_note = None
        # The previous widgets are bound to a command that may no longer exist.
        if self.panel is not None:
            self.panel.remove()
        self.panel = self.server.gui.add_folder("Selected command", order=3)
        cfg = self.commands[self.selected]
        with self.panel:
            # A gripper command holds its pose, so it has no goal to author.
            if isinstance(cfg, GripperCommand):
                self._create_gripper_fields()
            else:
                self._create_pose_fields(cfg)
        self.scrub.value = self.fraction
        self.sync_gizmos()

    def _create_gripper_fields(self) -> None:
        self._field("open_gripper", "Open gripper", boolean=True)
        self._field("duration_s", "Hold (s)", minimum=0.01)

    def _create_pose_fields(self, cfg) -> None:
        self._reference_ui(cfg)
        self._field("gripper_open", "Keep gripper open", boolean=True)
        self._field("angle_deg", "Angle (deg)")
        # The valve angle silently overrides the typed one.
        if isinstance(cfg, RotateFrameCfg) and cfg.use_valve_angle:
            self.server.gui.add_markdown(
                "Angle (deg) is unused while **Use remaining valve angle** is on. Edit Angle to switch to a fixed arc."
            )
        self._field("max_ee_rotation_deg", "Max EE rotation (deg)", minimum=0.0)
        self._field("axial_distance", "Screw travel (m)")
        self._field("axis", "Axis in reference frame", vector=True)
        self._field("use_valve_angle", "Use remaining valve angle", boolean=True)
        self._create_motion_folder()
        self._create_command_notes(cfg)

    def _create_motion_folder(self) -> None:
        with self.server.gui.add_folder("Motion settings", expand_by_default=False):
            for name, label, minimum in MOTION_FIELDS:
                self._field(name, label, minimum=minimum)
            self._field("canonicalize_upward", "Keep TCP upright", boolean=True)
            self._field("hold_current_orientation", "Hold starting orientation", boolean=True)

    def _create_command_notes(self, cfg) -> None:
        """Captions for authored state the widgets above cannot show."""
        gui = self.server.gui
        # The executed arc may come from the valve instead of Angle (deg).
        if isinstance(cfg, RotateFrameCfg):
            self.angle_note = gui.add_markdown(effective_arc_caption(cfg, self.segments[self.selected].handler))
        # A retargeted command follows a dense path that editing discards.
        if _reference_path(cfg) is not None:
            self.reference_note = gui.add_markdown(
                "Dense cuRobo reference is active. Editing this goal clears that reference."
            )
        override = _axis_position_override(cfg)
        # The pivot constraint is authored outside the editor and can only be dropped.
        if override is not None:
            gui.add_markdown(f"Base-frame constraint `axis_position_override_b`: `{override}`")
            self._button("Clear axis_position_override_b", partial(self.edit, "axis_position_override_b", None))

    def _reference_ui(self, cfg) -> None:
        """Frame the goal is anchored to, plus the pose fields of that anchor."""
        fixed = isinstance(cfg, GoToFrameCfg) and cfg.target_position_env is not None
        options = list(self.reference_options)
        # Only a GoTo can be pinned to the environment instead of a frame.
        if isinstance(cfg, GoToFrameCfg):
            options.insert(0, ENVIRONMENT_REFERENCE)
        current = ENVIRONMENT_REFERENCE if fixed else f"{cfg.frame_name}/{cfg.target_frame_name}"
        reference = self.server.gui.add_dropdown("Reference", options=options, initial_value=current)
        reference.on_update(self.events.on_change(self.change_reference, versioned=True))
        # A fixed goal is authored in environment coordinates, a frame goal as an offset.
        if fixed:
            self._create_fixed_goal_fields(cfg)
        else:
            self._field("target_offset_pos", "Offset in reference (m)", vector=True)
            self._field("target_offset_rot", "Rotation in reference (deg)", rotation=True)

    def _create_fixed_goal_fields(self, cfg) -> None:
        self._field("target_position_env", "Goal in environment (m)", vector=True)
        # Without an authored rotation the goal keeps the orientation from reset.
        if cfg.target_orientation_env is None:
            self.server.gui.add_markdown("Orientation holds its starting value. Dragging authors an explicit rotation.")
        else:
            self._field("target_orientation_env", "Goal rotation (deg)", rotation=True)

    def _field(self, name, label, *, boolean=False, vector=False, rotation=False, minimum=None) -> None:
        """Add the widget of one authored field, when the command carries it."""
        cfg = self.commands[self.selected]
        # Each command type exposes its own subset of the authored fields.
        if not hasattr(cfg, name):
            return
        value = getattr(cfg, name)
        gui = self.server.gui
        # Rotations are authored as Euler degrees and stored as quaternions.
        if rotation:
            value = self._euler_degrees(value)
        # The widget follows the kind of value the field holds.
        if boolean:
            handle = gui.add_checkbox(label, initial_value=value)
        elif vector or rotation:
            handle = gui.add_vector3(label, initial_value=value, step=1.0 if rotation else 0.001)
        else:
            handle = gui.add_number(label, initial_value=float(value), step=0.01, min=minimum)
        self.command_fields[name] = (handle, rotation)
        handle.on_update(self.events.on_change(partial(self._edit_field, name, rotation), versioned=True))

    def _edit_field(self, name: str, rotation: bool, value) -> None:
        self.edit(name, self._quat_from_degrees(value) if rotation else value)

    def _sync_command_fields(self) -> None:
        """Write authored values back into the panel widgets after a drag."""
        cfg = self.commands[self.selected]
        for name, (handle, rotation) in self.command_fields.items():
            value = getattr(cfg, name)
            handle.value = self._euler_degrees(value) if rotation else value

    # ------------------------------------------------------------- scene aids

    def sync_gizmos(self) -> None:
        """Point the gizmos and the rotation axis at the selected command."""
        cfg = self.commands[self.selected]
        segment = self.segments[self.selected]
        self.goal_gizmo.visible = not isinstance(cfg, GripperCommand) and not self.calibrate.value
        self.tcp_gizmo.visible = self.calibrate.value
        # A rotation is authored through its pivot frame rather than its end pose.
        pose = segment.pivot if isinstance(cfg, RotateFrameCfg) else segment.end
        self._update_pose(self.goal_gizmo, self.ik.to_world(*pose))
        self._update_pose(self.tcp_gizmo, self.ik.tcp_pose_w())
        self.events.remember_drag_pose("goal", self.goal_gizmo)
        self.events.remember_drag_pose("tcp", self.tcp_gizmo)
        self._draw_rotation_axis(segment)

    def _draw_rotation_axis(self, segment: Segment) -> None:
        """Draw the axis a rotation turns around; hide it for other commands."""
        rotating = isinstance(segment.cfg, RotateFrameCfg)
        # The handle always needs points, so other commands hide a degenerate line.
        if rotating:
            center = numpy(self.ik.to_world(*segment.pivot)[0])[0]
            axis = numpy(math_utils.quat_apply(tensor(self.ik.robot.data.root_quat_w), segment.handler.rot_axis_b))[0]
            points = np.array([[center - axis * 0.15, center + axis * 0.15]])
        else:
            points = np.zeros((1, 2, 3))
        self.server.scene.add_line_segments(
            "/editor/rotation_axis",
            points=points,
            colors=ROTATION_AXIS_COLOR,
            line_width=4.0,
            visible=rotating,
        )

    # --------------------------------------------------------------- previews

    def _plans_missing(self) -> bool:
        """True when a moved goal or edited field left a cuRobo segment without a plan."""
        return any(is_curobo_command(s.cfg) and _preview_plan(s.cfg) is None for s in self.segments)

    def preview(self, *, live: bool = False) -> None:
        """Place the robot at the current progress of the selected command.

        ``live`` marks a call mid-drag: a cuRobo replan is too slow to run every
        dragged frame, so a stale plan falls back to DLS until the gizmo settles.
        """
        segment = self.segments[self.selected]
        goal = segment.sample(self.fraction)
        # The robot must hold still while the TCP offset itself is being dragged.
        if self.calibrate.value:
            self.status.content = "**TCP calibration** — the robot stays still while you move the offset."
        elif self._plans_missing() and not live:
            # Re-solve the whole sequence rather than preview a stale or partial plan.
            self.plan_with_curobo()
            return
        else:
            self.status.content = self._preview_status(segment, goal, allow_dls_fallback=live)
        self._update_pose(self.actual_tcp, self.ik.tcp_pose_w())
        self._update_pose(self.desired_tcp, self.ik.to_world(*goal))
        self._refresh_notes(segment)
        self.sync_gizmos()

    def _preview_status(self, segment: Segment, goal, *, allow_dls_fallback: bool = False) -> str:
        start = time.perf_counter()
        pos_error, rot_error, source = self._move_robot_to(segment, goal, allow_dls_fallback=allow_dls_fallback)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return self._describe_residual(pos_error * 1000.0, rot_error, source, elapsed_ms)

    def _move_robot_to(self, segment: Segment, goal, *, allow_dls_fallback: bool = False) -> tuple[float, float, str]:
        """Write the joints reaching ``goal``; returns metre / degree residuals."""
        joints = segment.sample_joints(self.fraction)
        # Planned joints are authoritative: DLS could pick a different elbow branch.
        if joints is not None:
            q_arm, names = joints
            self.ik.apply_named_joints(names, q_arm, segment.gripper_open)
            return (*self.ik.measure(goal), "cuRobo joints")
        # Outside a live drag, preview() replans before reaching here; this is a bug backstop.
        if is_curobo_command(segment.cfg) and not allow_dls_fallback:
            raise RuntimeError("Selected command has no cuRobo plan.")
        return (*self.ik.solve(goal, segment.gripper_open, self.ik_iters), "DLS preview")

    def _describe_residual(self, pos_error_mm: float, rot_error_deg: float, source: str, elapsed_ms: float) -> str:
        """Markdown report of how closely the robot reached the previewed pose."""
        reached = pos_error_mm < REACHED_POSITION_MM and rot_error_deg < REACHED_ROTATION_DEG
        debug = self.ik.last_debug
        return (
            f"{'**Unsaved** · ' if self.dirty else ''}"
            f"{'Reached' if reached else 'Residual — inspect reach / joint limits'} · {source}\n\n"
            f"Position **{pos_error_mm:.1f} mm** · Rotation **{rot_error_deg:.1f}°** · {elapsed_ms:.0f} ms\n\n"
            f"TCP goal RPY {debug['tcp_goal_rpy_deg']}  +X {debug['tcp_goal_+X']}  +Z {debug['tcp_goal_+Z']}\n\n"
            f"TCP actual RPY {debug['tcp_actual_rpy_deg']}  +X {debug['tcp_actual_+X']}  "
            f"+Z {debug['tcp_actual_+Z']}\n\n"
            f"Flange actual RPY {debug['flange_actual_rpy_deg']}  +X {debug['flange_actual_+X']}"
        )

    def _refresh_notes(self, segment: Segment) -> None:
        """Update the captions that depend on the resolved geometry."""
        # Both captions only exist for the command types that can show them.
        if self.angle_note is not None:
            self.angle_note.content = effective_arc_caption(segment.cfg, segment.handler)
        if self.reference_note is not None:
            self.reference_note.visible = _reference_path(segment.cfg) is not None

    def plan_with_curobo(self, *, refresh_panel: bool = True) -> None:
        """Re-plan the whole sequence with the solver play.py uses."""
        self.playing = False
        self.status.content = "Planning with cuRobo…"
        invalidate_preview_plans_from(self.commands, 0)
        self.segments = build_segments(self.term, self.commands, self.ik.initial_pose_b())
        # Skipped while a value is typed, so the input keeps the keyboard focus.
        if refresh_panel:
            self.refresh_selected_panel()
        try:
            count = plan_curobo_segments(self.term, self.segments, self.ik)
        except Exception as err:
            # A partial plan chains onto stale poses, so drop all of it.
            invalidate_preview_plans_from(self.commands, 0)
            self.ik.reset()
            self.rebuild()
            self.status.content = format_editor_plan_failure(err)
            raise
        print(f"[EDITOR] cuRobo planned {count}/{len(self.segments)} segments", flush=True)
        self.rebuild()
        self.preview()

    def dump_ik_debug(self) -> None:
        """Print the full IK report of the pose currently previewed."""
        cfg = self.commands[self.selected]
        plan = _preview_plan(cfg)
        extra = {
            "command": f"{self.selected} {type(cfg).__name__}",
            "progress": round(float(self.fraction), 3),
            "gripper_open": self.segments[self.selected].gripper_open,
            "canonicalize_upward": cfg.canonicalize_upward if isinstance(cfg, GoToFrameCfg) else None,
            "preview": "curobo" if plan is not None else "dls",
            "plan_waypoints": int(plan["joints"].shape[0]) if plan is not None else 0,
        }
        text = format_ik_debug({**extra, **self.ik.last_debug})
        print(f"[EDITOR IK]\n{text}", flush=True)
        self.status.content = f"```\n{text}\n```"

    # ---------------------------------------------------------------- editing

    def edit(self, name: str, value) -> None:
        """Apply one authored field to a validated copy of the selected command."""
        candidate = copy.deepcopy(self.commands[self.selected])
        siblings_changed = apply_authored_command_field(candidate, name, value)
        # Moving the goal invalidates the dense path a retarget authored for it.
        if name in GOAL_FIELDS:
            self._clear_reference_path(candidate)
        validate_command(candidate)
        self._clear_preview_plan(candidate)
        self.commands[self.selected] = candidate
        self.changed(
            refresh_panel=siblings_changed or name == "axis_position_override_b",
            replan=name in REPLAN_FIELDS,
        )

    def changed(self, *, refresh_panel: bool = True, replan: bool = False, live: bool = False) -> None:
        """Redraw after an edit, re-planning when the planned geometry moved."""
        self.dirty, self.playing = True, False
        self.rebuild()
        # Planning already rebuilds the panel and previews the new pose.
        if replan and any(is_curobo_command(cmd) for cmd in self.commands):
            self.plan_with_curobo(refresh_panel=refresh_panel)
            return
        if refresh_panel:
            self.show_selected()
        else:
            # Preserve the input handles and keyboard focus while a value is typed.
            self.preview(live=live)

    def change_reference(self, choice: str) -> None:
        """Re-anchor the selected command, keeping its goal where it is now."""
        cfg = self.commands[self.selected]
        segment = self.segments[self.selected]
        pose = segment.pivot if isinstance(cfg, RotateFrameCfg) else segment.end
        pose_w = self.ik.to_world(*pose)
        # An environment goal keeps world values; a frame goal keeps an offset.
        if choice == ENVIRONMENT_REFERENCE:
            cfg.frame_name = cfg.target_frame_name = ""
            cfg.target_position_env = _values(pose_w[0] - self.env.scene.env_origins)
            cfg.target_orientation_env = _values(pose_w[1])
        else:
            cfg.frame_name, cfg.target_frame_name = self.reference_options[choice]
            # Only a GoTo has a fixed environment goal that the frame now replaces.
            if isinstance(cfg, GoToFrameCfg):
                cfg.target_position_env = cfg.target_orientation_env = None
            self._store_relative_pose(cfg, pose_w)
        self._clear_reference_path(cfg)
        self._clear_preview_plan(cfg)
        self.changed(replan=True)

    def _store_relative_pose(self, cfg, pose_w) -> None:
        """Write ``pose_w`` as the command's offset inside its reference frame."""
        sensor = self.env.scene[cfg.frame_name]
        index = sensor.data.target_frame_names.index(cfg.target_frame_name)
        offset = math_utils.subtract_frame_transforms(
            tensor(sensor.data.target_pos_w)[:, index],
            tensor(sensor.data.target_quat_w)[:, index],
            *pose_w,
        )
        cfg.target_offset_pos = _values(offset[0])
        cfg.target_offset_rot = _values(offset[1])

    def _clear_reference_path(self, cfg) -> None:
        """Forget the dense retargeted path so the authored goal leads again."""
        # Only a retargeted GoTo carries one.
        if _reference_path(cfg) is not None:
            cfg.reference_pos_env = None
            cfg.reference_quat_xyzw = None

    def _clear_preview_plan(self, cfg) -> None:
        """Drop the plan of ``cfg`` and of every command chained after it."""
        # The candidate of an in-flight edit is not part of the sequence yet.
        if hasattr(cfg, "_preview_plan"):
            cfg._preview_plan = None
        invalidate_preview_plans_from(self.commands, self.selected)

    def _edit_offset_rotation(self, rpy) -> None:
        self.edit_offset("rot", self._quat_from_degrees(rpy))

    def _apply_offset(self, offset) -> None:
        """Push a TCP offset to both the fast preview and the live command term.

        cuRobo plans read the offset from the term, not from PreviewIK, so a
        write here is what makes planning honor an edited offset at all.
        """
        self.ik.offset = offset
        self.term.set_body_offset(copy.deepcopy(offset))

    def edit_offset(self, name: str, value) -> None:
        """Apply a typed TCP offset, validated through the setup schema."""
        data = self._setup_payload()
        data["body_offset"][name] = list(value)
        _, offset = validate_setup(data)
        self._apply_offset(SequentialPoseCommandCfg.OffsetCfg(**offset))
        invalidate_preview_plans_from(self.commands, 0)
        # A full cuRobo replan on every keystroke would block the gizmo from moving live.
        self.changed(refresh_panel=False, live=True)

    def drag_tcp_offset(self, pose, phase: str) -> None:
        """Re-calibrate the tool offset from the dragged TCP gizmo."""
        self.playing = False
        pose_w = viser_to_pose(*pose, device=self.env.device)
        offset = math_utils.subtract_frame_transforms(
            tensor(self.ik.robot.data.body_pos_w)[:, self.ik.body_idx],
            tensor(self.ik.robot.data.body_quat_w)[:, self.ik.body_idx],
            *pose_w,
        )
        self._apply_offset(SequentialPoseCommandCfg.OffsetCfg(pos=_values(offset[0]), rot=_values(offset[1])))
        self.offset_pos.value = self.ik.offset.pos
        self.offset_rot.value = self._euler_degrees(self.ik.offset.rot)
        # The TCP offset feeds every cuRobo plan's goal pose, so all of them are now stale.
        invalidate_preview_plans_from(self.commands, 0)
        # calibrate.value is on while this fires, so preview() holds the robot still regardless.
        self._after_drag(refresh_panel=False, live=phase != "end")

    def drag_goal(self, pose, phase: str) -> None:
        """Move the goal, or the rotation pivot, of the selected command."""
        self.playing = False
        pose_w = viser_to_pose(*pose, device=self.env.device)
        cfg = self.commands[self.selected]
        added_orientation = False
        # A fixed goal stores world values; every other command stores an offset.
        if isinstance(cfg, GoToFrameCfg) and cfg.target_position_env is not None:
            added_orientation = cfg.target_orientation_env is None
            cfg.target_position_env = _values(pose_w[0] - self.env.scene.env_origins)
            cfg.target_orientation_env = _values(pose_w[1])
        else:
            self._store_relative_pose(cfg, pose_w)
        self._clear_reference_path(cfg)
        self._clear_preview_plan(cfg)
        # A goal that just gained an explicit rotation needs the matching field.
        self._after_drag(refresh_panel=added_orientation, live=phase != "end")

    def _after_drag(self, *, refresh_panel: bool, live: bool) -> None:
        """Redraw for the dragged pose, keeping the widgets of the live drag."""
        self.dirty = True
        self.rebuild()
        self._sync_command_fields()
        # Rebuilding the panel mid-drag would drop the gizmo's own widgets.
        if refresh_panel:
            self.show_selected()
        else:
            self.preview(live=live)

    # -------------------------------------------------------------- playback

    def set_calibrating(self, calibrating: bool) -> None:
        """Follow the TCP calibration checkbox; the gizmos read its state back."""
        self.playing = False
        self.sync_gizmos()
        self.preview()

    def seek(self, value: float) -> None:
        self.playing, self.fraction = False, float(value)
        self.preview()

    def pause(self) -> None:
        self.playing = False

    def play(self) -> None:
        """Replay the whole sequence from the first command."""
        self.playing = False
        self.calibrate.value = False
        self.ik.reset()
        self.selected, self.fraction = 0, 0.0
        # preview() plans automatically if a cuRobo command has none yet.
        self.rebuild()
        self.show_selected()
        self.playing = True

    def reset(self) -> None:
        """Put the robot back at the joint pose the environment reset to."""
        self.playing = False
        self.ik.reset()
        self.status.content = "Robot restored to its reset joint pose. Select a command to preview."
        self._update_pose(self.actual_tcp, self.ik.tcp_pose_w())
        self.sync_gizmos()

    def poll(self, dt: float) -> None:
        """Run one editor frame: queued interactions first, then playback."""
        self.events.drain()
        if self.playing:
            self._advance_playback(dt)

    def _advance_playback(self, dt: float) -> None:
        self.fraction = min(1.0, self.fraction + dt / self.segments[self.selected].duration)
        self.preview()
        # The command is complete; stop at the end of the sequence or roll on.
        if self.fraction >= 1.0:
            self._start_next_command()
        self.scrub.value = self.fraction

    def _start_next_command(self) -> None:
        # Playback stops at the last command instead of looping.
        if self.selected == len(self.commands) - 1:
            self.playing = False
            return
        self.selected += 1
        self.fraction = 0.0
        self.rebuild()
        self.show_selected()

    # ------------------------------------------------------------- save/load

    def _setup_payload(self) -> dict:
        return make_setup(self.task, self.term.cfg, self.commands, self.ik.offset)

    def save(self) -> None:
        self.out = Path(self.path.value).expanduser()
        save_setup(self.out, self._setup_payload())
        self.dirty = False
        self.status.content = f"Saved **{self.out}**. Load with `play.py --task {self.task} --setup {self.out}`."

    def reload(self) -> None:
        """Replace the sequence with the one stored in the setup file."""
        data = load_setup(Path(self.path.value).expanduser())
        # Validate task, robot and frame references against a disposable config.
        apply_setup(copy.deepcopy(self.env.cfg), data, task=self.task)
        commands, offset = validate_setup(data, task=self.task)
        self.commands = _curobo_commands(commands, body_name=self.term.cfg.body_name)
        self._apply_offset(SequentialPoseCommandCfg.OffsetCfg(**offset))
        self.offset_pos.value, self.offset_rot.value = self.ik.offset.pos, self._euler_degrees(self.ik.offset.rot)
        self.selected, self.dirty, self.playing = 0, False, False
        self.fraction = 1.0
        self.rebuild()
        # A loaded cuRobo sequence carries no plans, so build them before previewing.
        if any(is_curobo_command(cmd) for cmd in self.commands):
            self.plan_with_curobo()
        else:
            self.show_selected()


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--setup", help="Load an existing command setup JSON.")
    parser.add_argument("--out", help="Save path (default: --setup or configs/<task>.json).")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ik-iters", type=int, default=24)
    parser.add_argument("--smoke-test", action="store_true", help="Construct scene/UI, solve once and exit.")
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    # A preview needs at least one IK iteration to move the robot.
    if args.ik_iters < 1:
        parser.error("--ik-iters must be positive")
    # The editor needs Newton unless the command line already picked a physics preset.
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    args.visualizer = None
    args.visualizer_explicit = args.visualizer_disable_all = True
    return args, hydra_args


def _editor_env_cfg(args):
    """Task config with the editor's commands, prepared for a kinematic preview."""
    env_cfg, _ = resolve_task_config(args.task, "")
    pose_command = getattr(env_cfg.commands, "pose_command", None)
    # The editor authors sequential pose commands and nothing else.
    if not isinstance(pose_command, SequentialPoseCommandCfg):
        raise SystemExit("Choose a task with SequentialPoseCommandCfg (GoTo / gripper / Rotate).")
    # A loaded setup replaces the task's commands before they are upgraded.
    if args.setup:
        apply_setup(env_cfg, load_setup(args.setup), task=args.task)
    pose_command.commands = _curobo_commands(pose_command.commands, body_name=pose_command.body_name)
    _configure_preview(env_cfg, args)
    return env_cfg


def _configure_preview(env_cfg, args) -> None:
    """Strip everything the authoring preview does not run: extra envs, recording, physics."""
    env_cfg.scene.num_envs = 1
    env_cfg.recorders = None
    env_cfg.commands.pose_command.class_type = _PreviewCommand
    env_cfg.commands.pose_command.debug_vis = False
    env_cfg.commands.pose_command.path_debug_vis = False
    env_cfg.sim.visualizer_cfgs = None
    # Only FK / Jacobians are used: avoid creating the CUDA-only MJWarp solver.
    if args.device == "cpu":
        from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

        env_cfg.sim.physics = NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), use_cuda_graph=False)
        env_cfg.sim.device = "cpu"


def _open_viewer(args):
    """Viser viewer bound to the Newton model of the running simulation."""
    from isaaclab_newton.physics import NewtonManager
    from newton.viewer import ViewerViser

    viewer = ViewerViser(port=args.port, label="HiveBoard command setup")
    viewer.set_model(NewtonManager.get_model())
    return viewer


def _publish_frame(viewer, timestamp: float) -> None:
    """Send one frame of the current physics state to the browser."""
    from isaaclab_newton.physics import NewtonManager

    viewer.begin_frame(timestamp)
    viewer.log_state(NewtonManager.get_state())
    viewer.end_frame()


def _aim_camera(viewer, editor: CommandEditor) -> None:
    """Frame the robot's reset flange from a fixed three-quarter view."""
    center = numpy(editor.ik.initial_flange[0])[0]
    viewer._server.initial_camera.position = center + np.array([1.1, -1.1, 0.7])
    viewer._server.initial_camera.look_at = center


def _poll_editor(editor: CommandEditor, dt: float) -> None:
    """Run one editor frame; authoring errors belong in the status bar."""
    try:
        editor.poll(dt)
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        editor.playing = False
        editor.status.content = f"**Cannot apply edit:** {exc}"
        print(f"[EDITOR] {exc}", flush=True)


def _run_editor(env, viewer, args, out: Path) -> None:
    """Open the editor and drive the viewer until the browser disconnects."""
    editor = CommandEditor(env.unwrapped, viewer, args.task, out, args.ik_iters)
    _aim_camera(viewer, editor)
    print(f"[INFO] Command editor: {viewer.url}\n[INFO] Save to {out}", flush=True)
    previous = time.perf_counter()
    # Publish a complete frame before queued startup planning blocks poll().
    _publish_frame(viewer, previous)
    viewer._server.flush()
    while viewer.is_running():
        start = time.perf_counter()
        _poll_editor(editor, min(start - previous, 0.1))
        previous = start
        _publish_frame(viewer, start)
        # A smoke test only proves that the scene, the UI and one solve work.
        if args.smoke_test:
            editor.dump_ik_debug()
            print(f"[SMOKE] {len(editor.commands)} commands; {editor.status.content}", flush=True)
            return
        time.sleep(max(0, 1 / 30 - (time.perf_counter() - start)))


def main():
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg = _editor_env_cfg(args)
    out = Path(args.out) if args.out else Path(args.setup) if args.setup else default_setup_path(args.task)
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        viewer = None
        try:
            env.reset()
            refresh_frame_sensors(env.unwrapped)
            viewer = _open_viewer(args)
            _run_editor(env, viewer, args, out)
        except KeyboardInterrupt:
            pass
        finally:
            # Both the viewer thread and the simulation must close on any exit.
            if viewer is not None:
                viewer.close()
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
