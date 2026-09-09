# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp


@configclass
class SpotGainsTerminationsCfg:
    """Run the full clip; no valve success term."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
