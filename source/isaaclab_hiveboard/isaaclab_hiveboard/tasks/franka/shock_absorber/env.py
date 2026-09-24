# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 shock-absorber task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.shock_absorber.env import (
    AnymalShockAbsorberEnvCfg,
    AnymalShockAbsorberEnvCfg_PLAY,
)
from isaaclab_hiveboard.tasks.franka.common import shift_target_frames, use_franka
from isaaclab_hiveboard.tasks.scenes.shock_absorber import FRANKA_PIN_TIP_X, PIN_GRASP_FRAMES, PIN_GRASP_X

# The 2F-140 fingertips reach ~2 cm past its TCP; the FR3 TCP is the tip.
_PRESS_SHIFT = -0.020


def _use_franka_shock_absorber(env_cfg) -> None:
    use_franka(env_cfg, "shock_absorber", "reset_object_root")
    shift_target_frames(env_cfg, PIN_GRASP_FRAMES, FRANKA_PIN_TIP_X - PIN_GRASP_X)
    shift_target_frames(env_cfg, ("press",), _PRESS_SHIFT)


@configclass
class FrankaShockAbsorberEnvCfg(AnymalShockAbsorberEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard shock-absorber scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_shock_absorber(self)


@configclass
class FrankaShockAbsorberEnvCfg_PLAY(AnymalShockAbsorberEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_shock_absorber(self)
