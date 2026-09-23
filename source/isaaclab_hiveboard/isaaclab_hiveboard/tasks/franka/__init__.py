"""Franka tasks for HiveBoard."""

from .key.env import FrankaKeyEnvCfg, FrankaKeyEnvCfg_PLAY
from .drawer.env import FrankaDrawerEnvCfg, FrankaDrawerEnvCfg_PLAY
from .button.env import FrankaButtonEnvCfg, FrankaButtonEnvCfg_PLAY
from .ball_valve.env import FrankaBallValveEnvCfg, FrankaBallValveEnvCfg_PLAY
from .circuit_breaker.env import FrankaCircuitBreakerEnvCfg, FrankaCircuitBreakerEnvCfg_PLAY
from .high_torque_valve.env import FrankaHighTorqueValveEnvCfg, FrankaHighTorqueValveEnvCfg_PLAY
from .lamp.env import FrankaLampEnvCfg
from .small_valve.env import FrankaSmallValveEnvCfg, FrankaSmallValveEnvCfg_PLAY

__all__ = [
    "FrankaKeyEnvCfg",
    "FrankaKeyEnvCfg_PLAY",
    "FrankaDrawerEnvCfg",
    "FrankaDrawerEnvCfg_PLAY",
    "FrankaButtonEnvCfg",
    "FrankaButtonEnvCfg_PLAY",
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
