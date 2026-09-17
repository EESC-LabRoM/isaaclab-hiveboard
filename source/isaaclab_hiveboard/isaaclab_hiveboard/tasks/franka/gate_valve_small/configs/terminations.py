import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.progress import full_turn_success


@configclass
class TerminationsCfg:
    """Require actual stem rotation, including progress across every regrasp."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=full_turn_success,
        params={
            "command_name": "pose_command",
            "tolerance": math.radians(5.0),
        },
    )
