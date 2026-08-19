from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
