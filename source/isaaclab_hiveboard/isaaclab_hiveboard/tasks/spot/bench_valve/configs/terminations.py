# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import valve_rotation_success


@configclass
class BenchValveTerminationsCfg:
    """Episode ends on timeout or when the key sequence finishes with the valve at the stop."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=valve_rotation_success,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"]),
            "threshold_rad": math.radians(5.0),
        },
    )
