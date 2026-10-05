# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 high-torque-valve task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import CuroboPlannedRotateFrameCfg, GripperCommand
from isaaclab_hiveboard.tasks.anymal.high_torque_valve.env import AnymalHighTorqueValveEnvCfg, AnymalHighTorqueValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import FRANKA_SCENE_SHIFT, use_franka

# The handwheel stands ~0.25 m proud of the valve root, toward the robot, so
# the default shift leaves the approach TCP only 0.45 m from the FR3 base.
HIGH_TORQUE_SCENE_SHIFT = (FRANKA_SCENE_SHIFT[0] + 0.15, FRANKA_SCENE_SHIFT[1], FRANKA_SCENE_SHIFT[2])
# Time for the pads to clear the handwheel before the arm moves [s].
RELEASE_S = 0.4


def _use_franka_high_torque_valve(env_cfg) -> None:
    use_franka(env_cfg, "high_torque_valve", "reset_valve_root", HIGH_TORQUE_SCENE_SHIFT)
    # The shared sequence opens the hand during the retreat leg. The FR3 pads
    # open slower than the arm backs off, so they drag the light handwheel and
    # can spin it several turns past the goal. Let go first, then retreat.
    commands = env_cfg.commands.pose_command.commands
    rotate = next(i for i, c in enumerate(commands) if isinstance(c, CuroboPlannedRotateFrameCfg))
    commands.insert(rotate + 1, GripperCommand(open_gripper=True, duration_s=RELEASE_S))
    env_cfg.episode_length_s += RELEASE_S


@configclass
class FrankaHighTorqueValveEnvCfg(AnymalHighTorqueValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard high-torque-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_high_torque_valve(self)


@configclass
class FrankaHighTorqueValveEnvCfg_PLAY(AnymalHighTorqueValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_high_torque_valve(self)
