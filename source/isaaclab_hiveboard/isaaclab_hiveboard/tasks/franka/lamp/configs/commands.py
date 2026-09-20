from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, FRANKA_WORKSPACE, as_command_offset
from isaaclab_hiveboard.tasks.spot.lamp.configs.commands import FramePoseCommandsCfg as LampCommandsCfg


@configclass
class FramePoseCommandsCfg(LampCommandsCfg):
    """Seat the lamp with Franka using the shared sequence and TCP pose IK."""

    def __post_init__(self):
        self.pose_command.body_name = FRANKA_EE.body_name
        self.pose_command.body_offset = as_command_offset(FRANKA_EE)
        # Franka's base is on the table; the lamp is 0.4 m above it.
        # Spot's corresponding offsets are relative to its raised base.
        for command in self.pose_command.commands:
            for name in ("position_override_b", "axis_position_override_b"):
                override = getattr(command, name, None)
                if override is not None:
                    setattr(command, name, (override[0], override[1], FRANKA_WORKSPACE.object_pos[2] - 0.03))
