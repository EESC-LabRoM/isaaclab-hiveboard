# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot sliding-drawer task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.terminations import TerminationsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.env import AnymalDrawerEnvCfg, AnymalDrawerEnvCfg_PLAY
from isaaclab_hiveboard.tasks.spot.common import use_spot

PINCH_FRAMES = ("drawer_approaching", "drawer_grasp", "drawer_pulled")


@configclass
class SpotDrawerEnvCfg(AnymalDrawerEnvCfg):
    """Fixed-base Spot on the shared HiveBoard sliding-drawer scene."""

    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        use_spot(self, PINCH_FRAMES)


@configclass
class SpotDrawerEnvCfg_PLAY(AnymalDrawerEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        use_spot(self, PINCH_FRAMES)
