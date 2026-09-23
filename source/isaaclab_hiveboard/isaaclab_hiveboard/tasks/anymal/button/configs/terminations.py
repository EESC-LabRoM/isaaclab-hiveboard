# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import articulation_joint_ranges_success
from isaaclab_hiveboard.tasks.anymal.button.configs.commands import BUTTON_PRESSED


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_ranges_success,
        params={
            "command_name": "pose_command",
            "asset_name": "button",
            "ranges": {
                # Lid swung at least 10 deg past upright, clear of the button.
                "RevoluteJoint": (math.radians(10.0), math.radians(45.0)),
                # Pressed at least 8 of its 10 mm; a hard push overdrives the
                # stop by a few mm, so leave the lower bound open.
                "PrismaticJoint": (-0.02, BUTTON_PRESSED),
            },
        },
    )
