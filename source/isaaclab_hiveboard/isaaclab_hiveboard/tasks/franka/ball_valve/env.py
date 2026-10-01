# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 ball-valve task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import GripperCommand
from isaaclab_hiveboard.tasks.anymal.ball_valve.env import AnymalBallValveEnvCfg, AnymalBallValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka


def _use_franka_ball_valve(env_cfg) -> None:
    """Pinch across the lever's thickness and settle the fingers before turning."""
    use_franka(env_cfg, "ball_valve", "reset_valve_root")
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("approaching", "lever_pivot", "rotate_frame"):
            frame.offset = OffsetCfg(
                pos=frame.offset.pos,
                rot=(0.0, 0.7071068, 0.7071068, 0.0),
            )

    hand = env_cfg.actions.gripper_action
    env_cfg.actions.gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name=hand.asset_name,
        joint_names=hand.joint_names,
        open_command_expr=hand.open_command_expr,
        close_command_expr={"fr3_finger_joint.*": 0.010},
        close_speed=0.05,
    )
    for command in env_cfg.commands.pose_command.commands:
        if isinstance(command, GripperCommand) and not command.open_gripper:
            command.duration_s = max(command.duration_s, 0.7)


@configclass
class FrankaBallValveEnvCfg(AnymalBallValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard ball-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_ball_valve(self)


@configclass
class FrankaBallValveEnvCfg_PLAY(AnymalBallValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_ball_valve(self)
