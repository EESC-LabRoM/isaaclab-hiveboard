# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, ANYMAL_WORKSPACE, as_command_offset
from isaaclab_hiveboard.tasks.spot.lamp.configs.commands import FramePoseCommandsCfg as LampCommandsCfg


@configclass
class FramePoseCommandsCfg(LampCommandsCfg):
    """Seat the lamp with ANYmal + DynaArm using the shared sequence and TCP pose IK."""

    def __post_init__(self):
        self.pose_command.body_name = ANYMAL_EE.body_name
        self.pose_command.body_offset = as_command_offset(ANYMAL_EE)
        # ANYmal's base is raised on its legs; the lamp sits at ANYMAL_WORKSPACE
        # height above it. Spot's corresponding offsets are relative to its own
        # raised base.
        for command in self.pose_command.commands:
            override = getattr(command, "axis_position_override_b", None)
            if override is not None:
                command.axis_position_override_b = (override[0], override[1], ANYMAL_WORKSPACE.object_pos[2] - 0.03)
