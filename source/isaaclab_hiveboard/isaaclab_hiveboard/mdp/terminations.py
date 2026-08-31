# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def is_done(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Check if the task is successfully completed."""
    command: CommandTerm = env.command_manager.get_term(command_name)

    if not hasattr(command, "is_done"):
        raise AttributeError(
            f"The command term '{command_name}' does not have the method 'is_done'. "
            "Cannot use 'is_done' termination."
        )

    return command.is_done()


def valve_rotation_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg,
    threshold_rad: float,
) -> torch.Tensor:
    """Require sequence completion and proximity to the per-episode valve goal."""
    command: CommandTerm = env.command_manager.get_term(command_name)
    if not hasattr(command, "is_done"):
        raise AttributeError(
            f"The command term '{command_name}' does not have the method 'is_done'."
        )

    valve = env.scene[asset_cfg.name]
    # SceneEntityCfg collapses a selection covering every joint to ``slice(None)``.
    # Index the articulation first so this works for both slices and explicit lists.
    selected_joint_pos = valve.data.joint_pos[:, asset_cfg.joint_ids]
    if selected_joint_pos.shape[-1] != 1:
        raise ValueError(
            "valve_rotation_success requires exactly one selected valve joint; "
            f"received shape {tuple(selected_joint_pos.shape)}."
        )
    joint_angle = selected_joint_pos[:, 0]
    if not hasattr(command, "valve_joint_des"):
        raise AttributeError(
            f"The command term '{command_name}' has no per-episode valve goal."
        )
    return command.is_done() & (
        torch.abs(joint_angle - command.valve_joint_des) <= threshold_rad
    )


def button_task_success(
    env: ManagerBasedRLEnv,
    asset_name: str,
    lid_joint_name: str,
    button_joint_name: str,
    lid_open_pos: float,
    lid_open_threshold_rad: float,
    button_press_threshold: float,
) -> torch.Tensor:
    """Succeed when the cover is open and the button is depressed.

    The scripted pose sequence is only a heuristic; success is the HiveBoard
    stage-2 criterion on the two button joints, looked up by name.
    """
    asset = env.scene[asset_name]
    lid_ids, lid_found = asset.find_joints(lid_joint_name)
    button_ids, button_found = asset.find_joints(button_joint_name)
    if len(lid_ids) != 1 or len(button_ids) != 1:
        raise ValueError(
            f"Lid joint '{lid_joint_name}' -> {lid_found}, "
            f"button joint '{button_joint_name}' -> {button_found}, "
            f"available {list(asset.joint_names)}"
        )
    lid_angle = asset.data.joint_pos[:, lid_ids[0]]
    button_travel = asset.data.joint_pos[:, button_ids[0]]
    lid_open = torch.abs(lid_angle - lid_open_pos) <= lid_open_threshold_rad
    button_pressed = button_travel <= button_press_threshold
    return lid_open & button_pressed
def articulation_joint_position_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg,
    target: float,
    tolerance: float,
) -> torch.Tensor:
    """Require sequence completion and one articulation joint near a target."""
    command: CommandTerm = env.command_manager.get_term(command_name)
    if not hasattr(command, "is_done"):
        raise AttributeError(
            f"The command term '{command_name}' does not have the method 'is_done'."
        )

    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 1:
        raise ValueError(
            "articulation_joint_position_success requires exactly one joint; "
            f"received shape {tuple(joint_pos.shape)}."
        )
    return command.is_done() & (torch.abs(joint_pos[:, 0] - target) <= tolerance)
