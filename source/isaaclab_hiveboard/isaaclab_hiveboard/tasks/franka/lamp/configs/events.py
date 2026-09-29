from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.franka.common import finger_contact_stiffness
from isaaclab_hiveboard.tasks.spot.lamp.configs.events import LampEventCfg


@configclass
class FrankaLampEventCfg(LampEventCfg):
    """Reset Franka and the lamp with gravity compensation."""

    # Last, so it runs after the lamp's material events.
    finger_contacts = finger_contact_stiffness("Lamp")
