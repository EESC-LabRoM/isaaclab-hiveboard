from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import (
    is_done,
    valve_rotation_success,
)

# Angular tolerance on the valve's final angle, in radians.
#
# This was previously ``math.radians(0.010)``, which is 0.01 degrees - a
# 1.7e-4 rad band that the scripted expert never entered, so ``success`` never
# fired and demo collection wrote empty datasets. The scripted expert drives
# the valve to within 0.87-1.01 degrees (0.015-0.018 rad) of the goal and then
# holds there, so the tolerance has to sit above that band to be reachable at
# all. Two degrees leaves margin for the contact-rich final approach without
# accepting a visibly unturned valve.
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
