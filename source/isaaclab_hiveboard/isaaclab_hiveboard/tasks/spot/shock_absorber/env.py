# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot shock-absorber pin task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.shock_absorber.env import AnymalShockAbsorberEnvCfg, AnymalShockAbsorberEnvCfg_PLAY
from isaaclab_hiveboard.tasks.scenes.shock_absorber import PIN_PINCH_FRAMES
from isaaclab_hiveboard.tasks.spot.common import use_spot


@configclass
class SpotShockAbsorberEnvCfg(AnymalShockAbsorberEnvCfg):
    """Fixed-base Spot on the shared HiveBoard shock-absorber pin scene."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, PIN_PINCH_FRAMES)


@configclass
class SpotShockAbsorberEnvCfg_PLAY(AnymalShockAbsorberEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_spot(self, PIN_PINCH_FRAMES)
