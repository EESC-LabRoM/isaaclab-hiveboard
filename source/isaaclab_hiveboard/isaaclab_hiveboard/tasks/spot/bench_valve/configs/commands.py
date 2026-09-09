# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import (
    TRAJECTORY_JSON,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
)
from isaaclab_hiveboard.mdp.commands.joint_trajectory_command import JointTrajectoryCommandCfg


@configclass
class BenchValveCommandsCfg:
    """Fixed 50 Hz joint trajectory from the HiveBoard website Spot clip."""

    joint_command: JointTrajectoryCommandCfg = JointTrajectoryCommandCfg(
        trajectory_path=str(TRAJECTORY_JSON),
        command_dim=7,
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        valve_joint_closed=VALVE_JOINT_CLOSED,
        valve_joint_open=VALVE_JOINT_OPEN,
    )
