# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot M30 thread task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.m30_thread.env import AnymalM30ThreadEnvCfg, AnymalM30ThreadEnvCfg_PLAY
from isaaclab_hiveboard.tasks.scenes.screw import SCREW_PINCH_FRAMES
from isaaclab_hiveboard.tasks.spot.common import use_spot


@configclass
class SpotM30ThreadEnvCfg(AnymalM30ThreadEnvCfg):
    """Fixed-base Spot on the shared HiveBoard M30 thread scene."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, SCREW_PINCH_FRAMES)


@configclass
class SpotM30ThreadEnvCfg_PLAY(AnymalM30ThreadEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, SCREW_PINCH_FRAMES)
