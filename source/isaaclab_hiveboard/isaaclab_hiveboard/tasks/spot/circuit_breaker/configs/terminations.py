from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import command_done_term, is_done


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_done = command_done_term()
    success = DoneTerm(func=is_done, params={"command_name": "pose_command"})
