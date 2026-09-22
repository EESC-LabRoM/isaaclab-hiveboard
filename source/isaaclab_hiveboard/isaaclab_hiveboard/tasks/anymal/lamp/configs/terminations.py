# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import articulation_joint_position_success


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_position_success,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("lamp", joint_names=["PrismaticJoint"]),
            "target": 0.0,
            "tolerance": 0.004,
        },
    )
