# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Absolute TCP pose control for ANYmal + DynaArm's lamp task."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_ik_offset
from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_CLOSE,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.tasks.spot.lamp.configs.actions import SpotLampActionCfg


@configclass
class AnymalLampActionCfg(SpotLampActionCfg):
    def __post_init__(self):
        self.arm_action.joint_names = list(ANYMAL_ARM_JOINT_NAMES)
        self.arm_action.body_name = ANYMAL_EE.body_name
        self.arm_action.body_offset = as_ik_offset(ANYMAL_EE)
        self.gripper_action.joint_names = list(NEWTON_GRIPPER_JOINT_NAMES)
        self.gripper_action.open_command_expr = dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN))
        self.gripper_action.close_command_expr = dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_CLOSE))
