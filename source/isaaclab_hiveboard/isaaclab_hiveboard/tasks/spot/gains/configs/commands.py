# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Valve-free joint-space replay of the retargeted Spot arm clip."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import GAINS_TRAJECTORY_JSON, TCP_SITE_POS
from isaaclab_hiveboard.mdp.commands.joint_trajectory_command import JointTrajectoryCommandCfg


@configclass
class SpotGainsCommandsCfg:
    """Replay the 636-sample retargeted arm clip. No valve in the scene."""

    joint_command: JointTrajectoryCommandCfg = JointTrajectoryCommandCfg(
        trajectory_path=str(GAINS_TRAJECTORY_JSON),
        command_dim=7,
        valve_asset_name=None,
        ee_body_name="arm_link_wr1",
        ee_offset=TCP_SITE_POS,
        debug_vis=True,
    )
