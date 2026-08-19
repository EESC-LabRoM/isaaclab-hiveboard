# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import FrameTransformerData

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_name: str = "ee_frame",
) -> torch.Tensor:
    """Return the TCP pose in the robot base frame.

    Args:
        env: Manager-based environment.
        asset_cfg: Robot scene entity.
        frame_name: Frame transformer containing the TCP as its first target.

    Returns:
        TCP position and unique ``wxyz`` quaternion, shape ``(num_envs, 7)``.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    frame_data: FrameTransformerData = env.scene[frame_name].data
    position, orientation = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        frame_data.target_pos_w[:, 0],
        frame_data.target_quat_w[:, 0],
    )
    return torch.cat((position, math_utils.quat_unique(orientation)), dim=-1)


def object_root_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("ball_valve"),
) -> torch.Tensor:
    """Return an object's root pose in the robot base frame.

    Args:
        env: Manager-based environment.
        robot_cfg: Robot scene entity defining the base frame.
        object_cfg: Object scene entity whose root pose is observed.

    Returns:
        Object position and unique ``wxyz`` quaternion, shape ``(num_envs, 7)``.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: Articulation = env.scene[object_cfg.name]
    position, orientation = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((position, math_utils.quat_unique(orientation)), dim=-1)


def valve_task_direction(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """Return the valve task direction (+1 for opening, -1 for closing).

    Args:
        env: Manager-based environment.
        command_name: Name of the command term.

    Returns:
        Task direction of shape ``(num_envs, 1)``.
    """
    command = env.command_manager.get_term(command_name)
    return command.valve_task_goal.unsqueeze(-1)


def valve_current_angle(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "ball_valve", joint_names=["RevoluteJoint"]
    ),
) -> torch.Tensor:
    """Return the normalized current valve angle.

    Normalization maps the configured closed endpoint to zero and the open
    endpoint to one, including valves such as HiveBoard whose open angle is
    numerically smaller than its closed angle.

    Args:
        env: Manager-based environment.
        command_name: Name of the command term.
        asset_cfg: Scene entity configuration for the valve asset.

    Returns:
        Normalized current valve angle of shape ``(num_envs, 1)``.
    """
    command = env.command_manager.get_term(command_name)
    valve: Articulation = env.scene[asset_cfg.name]
    joint_pos = valve.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 1:
        raise ValueError("valve_current_angle requires exactly one selected valve joint")
    denominator = float(
        command.cfg.valve_joint_open - command.cfg.valve_joint_closed
    )
    if abs(denominator) < 1.0e-8:
        raise ValueError("Valve open and closed endpoints must differ")
    q_norm = (joint_pos[:, 0] - command.cfg.valve_joint_closed) / denominator
    return q_norm.unsqueeze(-1)


def valve_goal_angle(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
) -> torch.Tensor:
    """Return the normalized goal valve angle.

    Normalization maps the configured closed endpoint to zero and the open
    endpoint to one, including valves such as HiveBoard whose open angle is
    numerically smaller than its closed angle.

    Args:
        env: Manager-based environment.
        command_name: Name of the command term.

    Returns:
        Normalized goal valve angle of shape ``(num_envs, 1)``.
    """
    command = env.command_manager.get_term(command_name)
    denominator = float(
        command.cfg.valve_joint_open - command.cfg.valve_joint_closed
    )
    if abs(denominator) < 1.0e-8:
        raise ValueError("Valve open and closed endpoints must differ")
    q_des_norm = (
        command.valve_joint_des - command.cfg.valve_joint_closed
    ) / denominator
    return q_des_norm.unsqueeze(-1)


# Aliases
valve_direction = valve_task_direction
valve_task_current_angle = valve_current_angle
valve_task_goal = valve_goal_angle


def valve_task(
    env: ManagerBasedRLEnv,
    command_name: str = "pose_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "ball_valve", joint_names=["RevoluteJoint"]
    ),
) -> torch.Tensor:
    """Return the direction, normalized current angle, and normalized goal.

    Normalization maps the configured closed endpoint to zero and the open
    endpoint to one, including valves such as HiveBoard whose open angle is
    numerically smaller than its closed angle.
    """
    return torch.cat(
        (
            valve_task_direction(env, command_name=command_name),
            valve_current_angle(env, command_name=command_name, asset_cfg=asset_cfg),
            valve_goal_angle(env, command_name=command_name),
        ),
        dim=-1,
    )


def rel_ee_drawer_pose(
    env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data
    cabinet_tf_pose = cabinet_tf_data.target_pos_w[..., 0, :]
    cabinet_tf_quat = cabinet_tf_data.target_quat_w[..., 0, :]

    # print("Pose: ", cabinet_tf_data.target_pos_w[..., 0, :])

    asset: ArticulationData = env.scene[asset_cfg.name].data
    root_pos_w = asset.root_pos_w
    root_quat_w = asset.root_quat_w

    # compute the pose of the body in the root frame
    ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, cabinet_tf_pose, cabinet_tf_quat
    )

    return cabinet_tf_pose  # ee_pose_b


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[
        ..., 1:, :
    ] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
