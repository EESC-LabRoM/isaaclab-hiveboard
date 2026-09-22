# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_CLOSE,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    FLANGE_TO_TCP_POS,
    FLANGE_TO_TCP_QUAT_XYZW,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.mdp.pose_actions import OffsetDifferentialIKAction


@configclass
class AnymalBenchPositionActionCfg:
    """Absolute TCP pose targets and the editor's open/closed gripper poses."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        class_type=OffsetDifferentialIKAction,
        asset_name="robot",
        joint_names=list(ANYMAL_ARM_JOINT_NAMES),
        body_name="dynaarm_flange",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=FLANGE_TO_TCP_POS, rot=FLANGE_TO_TCP_QUAT_XYZW
        ),
        scale=1.0,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_CLOSE)),
    )
