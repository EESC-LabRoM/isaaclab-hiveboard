# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import button_task_success


@configclass
class TerminationsCfg:
    """Termination terms for the Spot hidden-button MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=button_task_success,
        params={
            "asset_name": "button",
            "lid_joint_name": "RevoluteJoint",
            "button_joint_name": "PrismaticJoint",
            "lid_open_pos": 0.0,
            "lid_open_threshold_rad": 0.75,
            "button_press_threshold": -0.004,
        },
    )
