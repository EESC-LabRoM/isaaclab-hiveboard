# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_CLOSE,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.mdp.actions.curobo_joint_action import (
    CuroboJointPositionActionCfg,
)

ANYMAL_ARM_6 = list(ANYMAL_ARM_JOINT_NAMES)


@configclass
class AnymalCuroboValveActionCfg:
    """DynaArm applies cuRobo joint plans as absolute positions; 2F-140 stays binary."""

    arm_action = CuroboJointPositionActionCfg(
        asset_name="robot",
        joint_names=ANYMAL_ARM_6,
        scale=1.0,
        preserve_order=True,
        command_name="pose_command",
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_CLOSE)),
    )
