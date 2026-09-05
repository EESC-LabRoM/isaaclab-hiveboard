# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot tasks on the HiveBoard benchmark."""

from .ball_valve.env import (
    SpotBallValveEnvCfg,
    SpotBallValveEnvCfg_PLAY,
    SpotBallValveEnvCfg_PLAY_CAMERAS,
)
from .button.env import SpotButtonEnvCfg
from .circuit_breaker.env import SpotCircuitBreakerEnvCfg
from .high_torque_valve.env import SpotHighTorqueValveEnvCfg
from .lamp.env import SpotLampEnvCfg
from .small_valve.env import SpotSmallValveEnvCfg

__all__ = [
    "SpotBallValveEnvCfg",
    "SpotBallValveEnvCfg_PLAY",
    "SpotBallValveEnvCfg_PLAY_CAMERAS",
    "SpotButtonEnvCfg",
    "SpotCircuitBreakerEnvCfg",
    "SpotHighTorqueValveEnvCfg",
    "SpotLampEnvCfg",
    "SpotSmallValveEnvCfg",
]
