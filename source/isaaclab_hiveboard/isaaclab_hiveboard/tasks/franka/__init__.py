"""Franka tasks for HiveBoard."""

from .ball_valve.env import FrankaBallValveEnvCfg, FrankaBallValveEnvCfg_PLAY
from .button.env import FrankaButtonEnvCfg, FrankaButtonEnvCfg_PLAY
from .circuit_breaker.env import FrankaCircuitBreakerEnvCfg, FrankaCircuitBreakerEnvCfg_PLAY
from .drawer.env import FrankaDrawerEnvCfg, FrankaDrawerEnvCfg_PLAY
from .high_torque_valve.env import FrankaHighTorqueValveEnvCfg, FrankaHighTorqueValveEnvCfg_PLAY
from .key.env import FrankaKeyEnvCfg, FrankaKeyEnvCfg_PLAY
from .lamp.env import FrankaLampEnvCfg
from .m8_thread.env import FrankaM8ThreadEnvCfg, FrankaM8ThreadEnvCfg_PLAY
from .m30_thread.env import FrankaM30ThreadEnvCfg, FrankaM30ThreadEnvCfg_PLAY
from .peg_insertion.env import FrankaPegInsertionEnvCfg, FrankaPegInsertionEnvCfg_PLAY
from .shock_absorber.env import FrankaShockAbsorberEnvCfg, FrankaShockAbsorberEnvCfg_PLAY
from .small_valve.env import FrankaSmallValveEnvCfg, FrankaSmallValveEnvCfg_PLAY

__all__ = [
    "FrankaKeyEnvCfg",
    "FrankaKeyEnvCfg_PLAY",
    "FrankaM8ThreadEnvCfg",
    "FrankaM8ThreadEnvCfg_PLAY",
    "FrankaM30ThreadEnvCfg",
    "FrankaM30ThreadEnvCfg_PLAY",
    "FrankaPegInsertionEnvCfg",
    "FrankaPegInsertionEnvCfg_PLAY",
    "FrankaShockAbsorberEnvCfg",
    "FrankaShockAbsorberEnvCfg_PLAY",
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
