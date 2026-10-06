from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.spot.lamp.configs.terminations import TerminationsCfg as LampTerminationsCfg


@configclass
class TerminationsCfg(LampTerminationsCfg):
    """Shared lamp terminations for Franka."""
