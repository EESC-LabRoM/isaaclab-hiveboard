# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_hiveboard.assets import (
    ANYMAL_EE,
    DYNAARM_JOINT_NAMES,
    ROBOTIQ_CLOSE_Q,
    ROBOTIQ_JOINT_GEAR,
    ROBOTIQ_OPEN_Q,
    as_ik_offset,
    robotiq_joint_targets,
)


@configclass
class AnymalIKAbsActionCfg:
    """Absolute TCP-pose IK plus a binary Robotiq 2F-140 command."""

    # ActionManager concatenates terms in declaration order. SequentialPoseCommand
    # emits [gripper, position XYZ, quaternion WXYZ], so the one-dimensional
    # gripper term must come before the three-dimensional arm term.
    # 1-D binary open/close; the expr maps onto finger_joint and inner fingers (second PD).
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ROBOTIQ_JOINT_GEAR),
        open_command_expr=robotiq_joint_targets(ROBOTIQ_OPEN_Q),
        close_command_expr=robotiq_joint_targets(ROBOTIQ_CLOSE_Q),
    )
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        # Position IK on the full DynaArm: extra DOF are needed to reach the
        # lever within a few centimetres. Orientation is not in the command.
        joint_names=list(DYNAARM_JOINT_NAMES),
        body_name=ANYMAL_EE.body_name,
        controller=DifferentialIKControllerCfg(
            # Full-pose DLS spins the DynaArm wrist (ori error 180 deg). Position
            # tracking is stable; the rotate phase still orbits the grasp point.
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        ),
        scale=1,
        body_offset=as_ik_offset(ANYMAL_EE),
    )
