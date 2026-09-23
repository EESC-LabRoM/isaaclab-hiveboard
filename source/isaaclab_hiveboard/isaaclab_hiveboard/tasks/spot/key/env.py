# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot lock-and-key task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.key.env import AnymalKeyEnvCfg, AnymalKeyEnvCfg_PLAY
from isaaclab_hiveboard.tasks.spot.common import use_spot

PINCH_FRAMES = ("key_approaching", "key_grasp")


@configclass
class SpotKeyEnvCfg(AnymalKeyEnvCfg):
    """Fixed-base Spot on the shared HiveBoard lock-and-key scene."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, PINCH_FRAMES)


@configclass
class SpotKeyEnvCfg_PLAY(AnymalKeyEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, PINCH_FRAMES)
