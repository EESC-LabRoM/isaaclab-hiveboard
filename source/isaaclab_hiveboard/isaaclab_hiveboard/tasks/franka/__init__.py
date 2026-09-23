"""Franka tasks for HiveBoard."""

from .ball_valve.env import FrankaBallValveEnvCfg, FrankaBallValveEnvCfg_PLAY
from .circuit_breaker.env import FrankaCircuitBreakerEnvCfg, FrankaCircuitBreakerEnvCfg_PLAY
from .high_torque_valve.env import FrankaHighTorqueValveEnvCfg, FrankaHighTorqueValveEnvCfg_PLAY
from .lamp.env import FrankaLampEnvCfg
from .small_valve.env import FrankaSmallValveEnvCfg, FrankaSmallValveEnvCfg_PLAY

__all__ = [
    "FrankaBallValveEnvCfg",
    "FrankaBallValveEnvCfg_PLAY",
    "FrankaCircuitBreakerEnvCfg",
    "FrankaCircuitBreakerEnvCfg_PLAY",
    "FrankaHighTorqueValveEnvCfg",
    "FrankaHighTorqueValveEnvCfg_PLAY",
    "FrankaLampEnvCfg",
    "FrankaSmallValveEnvCfg",
    "FrankaSmallValveEnvCfg_PLAY",
]
