from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp


@configclass
class TerminationsCfg:
    """Termination terms for the ANYmal ball-valve scene.

    This baseline has no scripted pose command, so the episode ends on timeout.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
