# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_CLOSE,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)


@configclass
class AnymalJointPositionActionCfg:
    """Absolute DynaArm joint positions from cuRobo waypoints, plus a binary 2F-140 gripper."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ANYMAL_ARM_JOINT_NAMES),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_CLOSE)),
    )
