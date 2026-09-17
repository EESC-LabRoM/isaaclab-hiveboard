from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.lever_valve.configs.events import FrankaLeverValveEventCfg


@configclass
class FrankaGateValveSmallEventCfg(FrankaLeverValveEventCfg):
    """Reset robot and passive valve to their defaults, including closed zero."""
