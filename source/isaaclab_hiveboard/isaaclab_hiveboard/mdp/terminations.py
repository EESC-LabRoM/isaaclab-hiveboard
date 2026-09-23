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
    selected_joint_pos = valve.data.joint_pos.torch[:, asset_cfg.joint_ids]
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
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if joint_pos.shape[-1] != 1:
        raise ValueError(
            "articulation_joint_position_success requires exactly one joint; "
            f"received shape {tuple(joint_pos.shape)}."
        )
    return command.is_done() & (torch.abs(joint_pos[:, 0] - target) <= tolerance)


def articulation_joint_ranges_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str,
    ranges: dict[str, tuple[float, float]],
) -> torch.Tensor:
    """Require sequence completion and every named joint inside its ``[low, high]`` range.

    Multi-stage mechanisms (e.g. the hidden button: lid open *and* button
    pressed) need more than one joint checked at once.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    if not hasattr(command, "is_done"):
        raise AttributeError(
            f"The command term '{command_name}' does not have the method 'is_done'."
        )

    asset = env.scene[asset_name]
    joint_pos = asset.data.joint_pos.torch
    success = command.is_done()
    for joint_name, (low, high) in ranges.items():
        joint_ids, _ = asset.find_joints(joint_name)
        if len(joint_ids) != 1:
            raise ValueError(f"Expected exactly one joint named '{joint_name}', found {len(joint_ids)}.")
        q = joint_pos[:, joint_ids[0]]
        success = success & (q >= low) & (q <= high)
    return success
