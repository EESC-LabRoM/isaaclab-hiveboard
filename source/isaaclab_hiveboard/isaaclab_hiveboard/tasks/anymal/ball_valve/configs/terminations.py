# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import is_done


@configclass
class TerminationsCfg:
    """End the episode when the open-valve sequence finishes or time runs out."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=is_done, params={"command_name": "pose_command"})
