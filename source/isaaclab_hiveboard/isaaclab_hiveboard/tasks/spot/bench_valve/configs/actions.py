# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES, TCP_SITE_POS
from isaaclab_hiveboard.mdp.pose_actions import OffsetDifferentialIKAction




@configclass
class SpotBenchJointActionCfg:
    """Absolute arm + gripper joint positions from the website clip."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class SpotBenchPositionActionCfg:
    """Absolute TCP pose targets and the editor's open/closed gripper poses."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        class_type=OffsetDifferentialIKAction,
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES[:-1]),
        body_name="arm_link_wr1",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_SITE_POS),
        scale=1.0,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.5},
        close_command_expr={"arm_f1x": -0.3},
    )
