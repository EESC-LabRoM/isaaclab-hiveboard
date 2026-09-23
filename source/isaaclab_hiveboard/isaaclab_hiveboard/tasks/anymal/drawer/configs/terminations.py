# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import articulation_joint_ranges_success
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_ranges_success,
        params={
            "command_name": "pose_command",
            "asset_name": "drawer",
            "ranges": {"PrismaticJoint": (DRAWER_OPEN, 1.0)},
        },
    )
