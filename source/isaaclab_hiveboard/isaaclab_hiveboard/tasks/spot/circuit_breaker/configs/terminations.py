from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import is_done


@configclass
class TerminationsCfg:
    """Termination terms for the Spot circuit-breaker MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=is_done, params={"command_name": "pose_command"})
