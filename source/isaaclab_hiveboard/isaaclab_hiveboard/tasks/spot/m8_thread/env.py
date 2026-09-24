# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot M8 thread task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.m8_thread.env import AnymalM8ThreadEnvCfg, AnymalM8ThreadEnvCfg_PLAY
from isaaclab_hiveboard.tasks.scenes.screw import SCREW_PINCH_FRAMES
from isaaclab_hiveboard.tasks.spot.common import use_spot


@configclass
class SpotM8ThreadEnvCfg(AnymalM8ThreadEnvCfg):
    """Fixed-base Spot on the shared HiveBoard M8 thread scene."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, SCREW_PINCH_FRAMES)


@configclass
class SpotM8ThreadEnvCfg_PLAY(AnymalM8ThreadEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, SCREW_PINCH_FRAMES)
