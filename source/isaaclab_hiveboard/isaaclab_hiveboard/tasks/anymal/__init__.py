# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal + DynaArm tasks on the HiveBoard benchmark."""

from .ball_valve.env import (
    AnymalBallValveEnvCfg,
    AnymalBallValveEnvCfg_PLAY,
)
from .bench_valve.env import (
    AnymalBenchValveEnvCfg,
    AnymalBenchValveEnvCfg_PLAY,
)
from .circuit_breaker.env import (
    AnymalCircuitBreakerEnvCfg,
    AnymalCircuitBreakerEnvCfg_PLAY,
)
from .curobo_valve.env import (
    AnymalCuroboValveEnvCfg,
    AnymalCuroboValveEnvCfg_PLAY,
)
from .high_torque_valve.env import (
    AnymalHighTorqueValveEnvCfg,
    AnymalHighTorqueValveEnvCfg_PLAY,
)
from .lamp.env import AnymalLampEnvCfg
from .small_valve.env import (
    AnymalSmallValveEnvCfg,
    AnymalSmallValveEnvCfg_PLAY,
)

__all__ = [
    "AnymalBallValveEnvCfg",
    "AnymalBallValveEnvCfg_PLAY",
    "AnymalBenchValveEnvCfg",
    "AnymalBenchValveEnvCfg_PLAY",
    "AnymalCircuitBreakerEnvCfg",
    "AnymalCircuitBreakerEnvCfg_PLAY",
    "AnymalCuroboValveEnvCfg",
    "AnymalCuroboValveEnvCfg_PLAY",
    "AnymalHighTorqueValveEnvCfg",
    "AnymalHighTorqueValveEnvCfg_PLAY",
    "AnymalLampEnvCfg",
    "AnymalSmallValveEnvCfg",
    "AnymalSmallValveEnvCfg_PLAY",
]
