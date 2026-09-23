# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot hidden-button task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.button.env import AnymalButtonEnvCfg, AnymalButtonEnvCfg_PLAY
from isaaclab_hiveboard.tasks.spot.common import use_spot

LID_PINCH_FRAMES = ("lid_approaching", "lid_grasp")


@configclass
class SpotButtonEnvCfg(AnymalButtonEnvCfg):
    """Fixed-base Spot on the shared HiveBoard hidden-button scene."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, LID_PINCH_FRAMES)


@configclass
class SpotButtonEnvCfg_PLAY(AnymalButtonEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, LID_PINCH_FRAMES)
