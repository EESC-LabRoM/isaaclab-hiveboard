from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.spot.lamp.configs.events import LampEventCfg


@configclass
class FrankaLampEventCfg(LampEventCfg):
    """Reset Franka and the lamp with gravity compensation."""
