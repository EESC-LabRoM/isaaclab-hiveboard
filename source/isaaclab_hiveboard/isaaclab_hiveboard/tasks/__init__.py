# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task registrations for HiveBoard multi-robot manipulation environments."""

import gymnasium as gym

##
# Spot Environments
##

gym.register(
    id="validate_command_spot",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.validate_command_spot.env:ValidateCommandSpotEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-BallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.env:SpotBallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-BallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.env:SpotBallValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-CircuitBreaker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.circuit_breaker.env:SpotCircuitBreakerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-CircuitBreaker-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.circuit_breaker.env:SpotCircuitBreakerEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-HighTorqueValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.high_torque_valve.env:SpotHighTorqueValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-HighTorqueValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.high_torque_valve.env:SpotHighTorqueValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-SmallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.small_valve.env:SpotSmallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-SmallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.small_valve.env:SpotSmallValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-BenchValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.bench_valve.env:SpotBenchValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-BenchValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.bench_valve.env:SpotBenchValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-CuroboValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.curobo_valve.env:SpotCuroboValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-CuroboValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.curobo_valve.env:SpotCuroboValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Gains-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.gains.env:SpotGainsEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Gains-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.gains.env:SpotGainsEnvCfg_PLAY",
    },
)

##
# ANYmal Environments
##

gym.register(
    id="Isaac-HiveBoard-Anymal-CuroboValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.curobo_valve.env:AnymalCuroboValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-CuroboValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.curobo_valve.env:AnymalCuroboValveEnvCfg_PLAY",
    },
)
