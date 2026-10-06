from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cabinet import mdp

from isaaclab_hiveboard.mdp.terminations import command_done_term, screw_travel_success


@configclass
class TerminationsCfg:
    """Success once the sequence ends with the bulb screwed in as far as it commands."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    command_done = command_done_term()
    success = DoneTerm(
        func=screw_travel_success,
        # A third of a quarter turn at the 6 mm/revolution pitch.
        params={"command_name": "pose_command", "tolerance": 0.0005},
    )
