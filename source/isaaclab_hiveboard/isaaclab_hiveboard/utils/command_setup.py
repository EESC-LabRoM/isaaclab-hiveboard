# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable command-editor settings, applied before constructing an environment.

Only data from the explicitly supported command configs is serialized. Loading a
setup never imports a class named in the file. All rotations are xyzw.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import tempfile
from pathlib import Path

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    SequentialPoseCommandCfg,
)

COMMAND_TYPES = {
    cls.__name__: cls
    for cls in (
        GoToFrameCfg,
        GripperCommand,
        RotateFrameCfg,
        ScrewFrameCfg,
        CuroboPlannedGoToFrameCfg,
        CuroboPlannedRotateFrameCfg,
    )
}


def _vector(value, size: int, name: str, *, quaternion: bool = False) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain finite numbers")
    result = tuple(float(v) for v in value)
    if quaternion:
        norm = math.sqrt(sum(v * v for v in result))
        if norm < 1e-8:
            raise ValueError(f"{name} must be a non-zero quaternion")
        result = tuple(v / norm for v in result)
    return result


def validate_command(cmd) -> None:
    """Validate editable values before they reach a handler or an IK solver."""
    if type(cmd).__name__ not in COMMAND_TYPES:
        raise ValueError(f"Unsupported command: {type(cmd).__name__}")
    for name in ("duration_s", "velocity", "angular_velocity"):
        if hasattr(cmd, name):
            value = getattr(cmd, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
    for name in ("distance_threshold", "orientation_threshold_deg", "angle_threshold_deg", "max_ee_rotation_deg"):
        if hasattr(cmd, name):
            value = getattr(cmd, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
    for name in ("angle_deg", "axial_distance"):
        if hasattr(cmd, name) and not math.isfinite(getattr(cmd, name)):
            raise ValueError(f"{name} must be finite")
    for name in ("gripper_open", "open_gripper", "use_valve_angle", "canonicalize_upward", "hold_current_orientation"):
        if hasattr(cmd, name) and not isinstance(getattr(cmd, name), bool):
            raise ValueError(f"{name} must be a boolean")
    for name in ("target_offset_pos", "target_position_env", "axis"):
        if getattr(cmd, name, None) is not None:
            setattr(cmd, name, _vector(getattr(cmd, name), 3, name))
    for name in ("target_offset_rot", "target_orientation_env"):
        if getattr(cmd, name, None) is not None:
            setattr(cmd, name, _vector(getattr(cmd, name), 4, name, quaternion=True))
    if isinstance(cmd, RotateFrameCfg) and sum(v * v for v in cmd.axis) < 1e-12:
        raise ValueError("Rotation axis must be non-zero")
    for name in ("position_override_b", "axis_position_override_b"):
        value = getattr(cmd, name, None)
        if value is not None:
            _vector([0 if v is None else v for v in value], 3, name)
    if isinstance(cmd, (GoToFrameCfg, RotateFrameCfg)) and getattr(cmd, "target_position_env", None) is None:
        if not isinstance(cmd.frame_name, str) or not cmd.frame_name:
            raise ValueError("A frame-relative command needs a frame sensor")
        if not isinstance(cmd.target_frame_name, str) or not cmd.target_frame_name:
            raise ValueError("A frame-relative command needs a target frame")


PLANNER_FIELDS = (
    "robot_joint_names",
    "robot_curobo_yaml",
    "robot_urdf",
    "num_ik_seeds",
    "num_trajopt_seeds",
    "max_plan_attempts",
    "interpolation_buffer_size",
    "max_joint_step",
)


def _params(cmd, cls) -> dict:
    names = {field.name for field in dataclasses.fields(cls)} - {"class_type"}
    return {name: copy.deepcopy(getattr(cmd, name)) for name in names if hasattr(cmd, name)}


def is_curobo_command(cmd) -> bool:
    return isinstance(cmd, (CuroboPlannedGoToFrameCfg, CuroboPlannedRotateFrameCfg))


def planner_settings(commands) -> dict:
    """Robot / solver fields copied from the first cuRobo command in ``commands``."""
    for cmd in commands:
        if not is_curobo_command(cmd):
            continue
        settings = {}
        for name in PLANNER_FIELDS:
            if hasattr(cmd, name):
                settings[name] = copy.deepcopy(getattr(cmd, name))
        if settings.get("robot_joint_names"):
            return settings
    raise ValueError(
        "No cuRobo planner settings on this sequence. Insert or convert after a CuroboPlanned command, "
        "or start from a task that already uses cuRobo."
    )


def as_curobo_command(cmd, planner: dict):
    """Keep Cartesian fields; attach cuRobo planner settings."""
    if is_curobo_command(cmd):
        return copy.deepcopy(cmd)
    if isinstance(cmd, GoToFrameCfg):
        cls, params = CuroboPlannedGoToFrameCfg, _params(cmd, GoToFrameCfg)
    elif isinstance(cmd, RotateFrameCfg) and not isinstance(cmd, ScrewFrameCfg):
        cls, params = CuroboPlannedRotateFrameCfg, _params(cmd, RotateFrameCfg)
    else:
        raise ValueError(f"Cannot use a cuRobo plan for {type(cmd).__name__}")
    allowed = {field.name for field in dataclasses.fields(cls)} - {"class_type"}
    params.update({name: copy.deepcopy(value) for name, value in planner.items() if name in allowed})
    result = cls(**params)
    validate_command(result)
    return result


def as_direct_command(cmd):
    """Drop planner fields and keep the Cartesian GoTo / Rotate command."""
    if isinstance(cmd, CuroboPlannedGoToFrameCfg):
        result = GoToFrameCfg(**_params(cmd, GoToFrameCfg))
    elif isinstance(cmd, CuroboPlannedRotateFrameCfg):
        result = RotateFrameCfg(**_params(cmd, RotateFrameCfg))
    else:
        return copy.deepcopy(cmd)
    validate_command(result)
    return result


def encode_command(cmd) -> dict:
    validate_command(cmd)
    return {
        "type": type(cmd).__name__,
        "parameters": {
            field.name: copy.deepcopy(getattr(cmd, field.name))
            for field in dataclasses.fields(cmd)
            if field.name != "class_type"
        },
    }


def decode_command(data: dict):
    if not isinstance(data, dict) or set(data) != {"type", "parameters"}:
        raise ValueError("Each command needs type and parameters")
    cls = COMMAND_TYPES.get(data["type"])
    if cls is None:
        raise ValueError(f"Unsupported command type: {data['type']!r}")
    allowed = {f.name for f in dataclasses.fields(cls)} - {"class_type"}
    params = data["parameters"]
    if not isinstance(params, dict) or set(params) - allowed:
        raise ValueError(f"Unknown parameters for {cls.__name__}")
    cmd = cls(**copy.deepcopy(params))
    validate_command(cmd)
    return cmd


def make_setup(task: str, command_cfg, commands=None, offset=None) -> dict:
    if not isinstance(command_cfg, SequentialPoseCommandCfg):
        raise ValueError("The editor requires a SequentialPoseCommandCfg")
    offset = offset or command_cfg.body_offset or SequentialPoseCommandCfg.OffsetCfg()
    return {
        "version": 1,
        "task": task,
        "asset_name": command_cfg.asset_name,
        "body_name": command_cfg.body_name,
        "body_offset": {"pos": list(offset.pos), "rot": list(offset.rot)},
        "commands": [encode_command(c) for c in (command_cfg.commands if commands is None else commands)],
    }


def validate_setup(data: dict, *, task: str | None = None) -> tuple[list, dict]:
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Expected command setup version 1")
    for name in ("task", "asset_name", "body_name"):
        if not isinstance(data.get(name), str) or not data[name]:
            raise ValueError(f"Setup needs {name}")
    if task is not None and data["task"] != task:
        raise ValueError(f"Setup belongs to {data['task']!r}, not {task!r}")
    offset = data.get("body_offset")
    if not isinstance(offset, dict) or set(offset) != {"pos", "rot"}:
        raise ValueError("body_offset needs pos and rot (xyzw)")
    offset = {
        "pos": _vector(offset["pos"], 3, "body_offset.pos"),
        "rot": _vector(offset["rot"], 4, "body_offset.rot", quaternion=True),
    }
    if not isinstance(data.get("commands"), list) or not data["commands"]:
        raise ValueError("A setup must contain at least one command")
    return [decode_command(cmd) for cmd in data["commands"]], offset


def load_setup(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    validate_setup(data)
    return data


def save_setup(path: str | Path, data: dict) -> None:
    """Replace the settings atomically so interrupted saves remain readable."""
    validate_setup(data)
    content = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def apply_setup(env_cfg, data: dict, *, task: str) -> None:
    """Apply command, IK-action and TCP-sensor offsets before gym.make()."""
    commands, offset = validate_setup(data, task=task)
    cfg = getattr(env_cfg.commands, "pose_command", None)
    if not isinstance(cfg, SequentialPoseCommandCfg):
        raise ValueError("Setup requires a sequential pose_command task")
    if (cfg.asset_name, cfg.body_name) != (data["asset_name"], data["body_name"]):
        raise ValueError("Setup robot / end-effector body does not match the task")
    # Validate all references before making any changes.
    for cmd in commands:
        if isinstance(cmd, GripperCommand) or getattr(cmd, "target_position_env", None) is not None:
            continue
        sensor = getattr(env_cfg.scene, cmd.frame_name, None)
        names = [frame.name for frame in getattr(sensor, "target_frames", ())]
        if cmd.target_frame_name not in names:
            raise ValueError(f"Unknown reference {cmd.frame_name}/{cmd.target_frame_name}")
    cfg.commands = commands
    cfg.body_offset = SequentialPoseCommandCfg.OffsetCfg(**offset)
    for action in vars(env_cfg.actions).values():
        if (
            getattr(action, "asset_name", None) == cfg.asset_name
            and getattr(action, "body_name", None) == cfg.body_name
        ):
            if hasattr(action, "body_offset"):
                from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

                action.body_offset = DifferentialInverseKinematicsActionCfg.OffsetCfg(**offset)
    ee_frame = getattr(env_cfg.scene, "ee_frame", None)
    if ee_frame is not None:
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

        for frame in ee_frame.target_frames:
            if frame.name == "ee_tcp":
                frame.offset = OffsetCfg(**offset)
