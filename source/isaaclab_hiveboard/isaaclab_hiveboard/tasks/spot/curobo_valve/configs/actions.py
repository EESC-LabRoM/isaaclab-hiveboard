# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES
from isaaclab_hiveboard.mdp.actions.curobo_joint_action import (
    CuroboJointPositionActionCfg,
)

SPOT_ARM_6 = list(ARM_JOINT_NAMES[:-1])


@configclass
class SpotCuroboValveActionCfg:
    """Arm applies cuRobo joint plans as absolute positions; gripper stays binary."""

    arm_action = CuroboJointPositionActionCfg(
        asset_name="robot",
        joint_names=SPOT_ARM_6,
        scale=1.0,
        preserve_order=True,
        command_name="pose_command",
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.5},
        close_command_expr={"arm_f1x": -0.3},
    )
