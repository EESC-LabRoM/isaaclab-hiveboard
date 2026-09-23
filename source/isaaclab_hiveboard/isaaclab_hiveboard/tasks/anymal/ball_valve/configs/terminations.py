# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import (
    is_done,
    valve_rotation_success,
)

# Angular tolerance on the valve's final angle, in radians. Same value and same
# reasoning as the Spot ball valve (see ``tasks/spot/ball_valve/configs/
# terminations.py``): this was ``math.radians(0.010)``, a 1.7e-4 rad band the
# scripted expert never enters, so ``success`` never fired and demo collection
# wrote empty datasets. Two degrees sits above the ~1 degree the cuRobo
# sequence actually achieves, without accepting a visibly unturned valve.
VALVE_SUCCESS_TOLERANCE_RAD = 0.035


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=is_done, params=dict({"command_name": "pose_command"}))


@configclass
class DeltaCollectionTerminationsCfg(TerminationsCfg):
    """Collection success requires reaching the sampled open/close endpoint."""

    success = DoneTerm(
        func=valve_rotation_success,
        params={
            "command_name": "pose_command",
            "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"]),
            "threshold_rad": VALVE_SUCCESS_TOLERANCE_RAD,
        },
    )
