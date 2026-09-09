# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES


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
