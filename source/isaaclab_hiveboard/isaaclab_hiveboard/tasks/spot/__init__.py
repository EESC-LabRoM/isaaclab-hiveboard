# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot tasks on the HiveBoard benchmark."""

from .key.env import SpotKeyEnvCfg, SpotKeyEnvCfg_PLAY
from .drawer.env import SpotDrawerEnvCfg, SpotDrawerEnvCfg_PLAY
from .button.env import SpotButtonEnvCfg, SpotButtonEnvCfg_PLAY
from .ball_valve.env import (
    SpotBallValveEnvCfg,
    SpotBallValveEnvCfg_PLAY,
)
from .circuit_breaker.env import (
    SpotCircuitBreakerEnvCfg,
    SpotCircuitBreakerEnvCfg_PLAY,
)
from .bench_valve.env import (
    SpotBenchValveEnvCfg,
    SpotBenchValveEnvCfg_PLAY,
)
from .curobo_valve.env import (
    SpotCuroboValveEnvCfg,
    SpotCuroboValveEnvCfg_PLAY,
)
from .gains.env import (
    SpotGainsEnvCfg,
    SpotGainsEnvCfg_PLAY,
)
from .high_torque_valve.env import (
    SpotHighTorqueValveEnvCfg,
    SpotHighTorqueValveEnvCfg_PLAY,
)
from .lamp.env import SpotLampEnvCfg
from .small_valve.env import (
    SpotSmallValveEnvCfg,
    SpotSmallValveEnvCfg_PLAY,
)
__all__ = [
    "SpotKeyEnvCfg",
    "SpotKeyEnvCfg_PLAY",
    "SpotDrawerEnvCfg",
    "SpotDrawerEnvCfg_PLAY",
    "SpotButtonEnvCfg",
    "SpotButtonEnvCfg_PLAY",
    "SpotBallValveEnvCfg",
    "SpotBallValveEnvCfg_PLAY",
    "SpotCircuitBreakerEnvCfg",
    "SpotCircuitBreakerEnvCfg_PLAY",
    "SpotBenchValveEnvCfg",
    "SpotBenchValveEnvCfg_PLAY",
    "SpotCuroboValveEnvCfg",
    "SpotCuroboValveEnvCfg_PLAY",
    "SpotGainsEnvCfg",
    "SpotGainsEnvCfg_PLAY",
    "SpotHighTorqueValveEnvCfg",
    "SpotHighTorqueValveEnvCfg_PLAY",
    "SpotLampEnvCfg",
    "SpotSmallValveEnvCfg",
    "SpotSmallValveEnvCfg_PLAY",
]
