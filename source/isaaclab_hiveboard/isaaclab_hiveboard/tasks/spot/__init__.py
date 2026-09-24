# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot tasks on the HiveBoard benchmark."""

from .ball_valve.env import (
    SpotBallValveEnvCfg,
    SpotBallValveEnvCfg_PLAY,
)
from .bench_valve.env import (
    SpotBenchValveEnvCfg,
    SpotBenchValveEnvCfg_PLAY,
)
from .button.env import SpotButtonEnvCfg, SpotButtonEnvCfg_PLAY
from .circuit_breaker.env import (
    SpotCircuitBreakerEnvCfg,
    SpotCircuitBreakerEnvCfg_PLAY,
)
from .curobo_valve.env import (
    SpotCuroboValveEnvCfg,
    SpotCuroboValveEnvCfg_PLAY,
)
from .drawer.env import SpotDrawerEnvCfg, SpotDrawerEnvCfg_PLAY
from .gains.env import (
    SpotGainsEnvCfg,
    SpotGainsEnvCfg_PLAY,
)
from .high_torque_valve.env import (
    SpotHighTorqueValveEnvCfg,
    SpotHighTorqueValveEnvCfg_PLAY,
)
from .key.env import SpotKeyEnvCfg, SpotKeyEnvCfg_PLAY
from .lamp.env import SpotLampEnvCfg
from .m8_thread.env import SpotM8ThreadEnvCfg, SpotM8ThreadEnvCfg_PLAY
from .m30_thread.env import SpotM30ThreadEnvCfg, SpotM30ThreadEnvCfg_PLAY
from .peg_insertion.env import SpotPegInsertionEnvCfg, SpotPegInsertionEnvCfg_PLAY
from .shock_absorber.env import SpotShockAbsorberEnvCfg, SpotShockAbsorberEnvCfg_PLAY
from .small_valve.env import (
    SpotSmallValveEnvCfg,
    SpotSmallValveEnvCfg_PLAY,
)

__all__ = [
    "SpotKeyEnvCfg",
    "SpotKeyEnvCfg_PLAY",
    "SpotM8ThreadEnvCfg",
    "SpotM8ThreadEnvCfg_PLAY",
    "SpotM30ThreadEnvCfg",
    "SpotM30ThreadEnvCfg_PLAY",
    "SpotPegInsertionEnvCfg",
    "SpotPegInsertionEnvCfg_PLAY",
    "SpotShockAbsorberEnvCfg",
    "SpotShockAbsorberEnvCfg_PLAY",
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
