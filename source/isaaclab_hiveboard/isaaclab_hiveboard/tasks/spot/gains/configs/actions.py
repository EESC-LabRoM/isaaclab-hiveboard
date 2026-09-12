# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gains validation actions: continuous arm joints plus a binary open/close gripper."""

from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES

GRIPPER_OPEN = -1.5
GRIPPER_CLOSE = -0.3
# Midpoint of the clip's gripper ramp; matches the snapped gains trajectory.
GRIPPER_THRESHOLD = -0.9


@configclass
class SpotGainsActionCfg:
    """6 continuous arm joints; the clip's gripper snaps to open/close targets."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES[:-1]),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper_action = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": GRIPPER_OPEN},
        close_command_expr={"arm_f1x": GRIPPER_CLOSE},
        threshold=GRIPPER_THRESHOLD,
        # Open values sit below close values on this gripper.
        positive_threshold=False,
    )
