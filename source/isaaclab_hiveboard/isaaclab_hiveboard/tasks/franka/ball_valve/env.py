# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 ball-valve task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.ball_valve.env import AnymalBallValveEnvCfg, AnymalBallValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka


@configclass
class FrankaBallValveEnvCfg(AnymalBallValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard ball-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        use_franka(self, "ball_valve", "reset_valve_root")


@configclass
class FrankaBallValveEnvCfg_PLAY(AnymalBallValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        use_franka(self, "ball_valve", "reset_valve_root")
