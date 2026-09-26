# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import command_done_term
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN
from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_slide_success


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_done = command_done_term()
    success = DoneTerm(
        func=drawer_slide_success,
        params={
            "command_name": "pose_command",
            "box_name": "drawer",
            "housing_name": "drawer_housing",
            "open_distance": DRAWER_OPEN,
        },
    )
