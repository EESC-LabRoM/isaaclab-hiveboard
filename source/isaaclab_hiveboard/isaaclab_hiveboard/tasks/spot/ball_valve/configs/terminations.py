import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import (
    is_done,
    valve_rotation_success,
)


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
            "threshold_rad": math.radians(15.0),
        },
    )
