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
    id="Isaac-HiveBoard-Spot-Lamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.lamp.env:SpotLampEnvCfg",
        # Consumed by Isaac Lab's scripts/imitation_learning/robomimic/train.py
        # and by scripts/imitation/train_bc.py.
        "robomimic_bc_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.lamp.agents:robomimic/bc.json",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Lamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.lamp.env:FrankaLampEnvCfg"},
)

gym.register(
    id="Spot-Manipulation-Lamp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.lamp.env:SpotLampEnvCfg"},
)

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
        "robomimic_bc_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.agents:robomimic/bc.json",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-BallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.env:SpotBallValveEnvCfg_PLAY",
        "robomimic_bc_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.agents:robomimic/bc.json",
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

gym.register(
    id="Isaac-HiveBoard-Anymal-Lamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.lamp.env:AnymalLampEnvCfg"},
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.env:AnymalBallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.env:AnymalBallValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-CircuitBreaker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.circuit_breaker.env:AnymalCircuitBreakerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-CircuitBreaker-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.circuit_breaker.env:AnymalCircuitBreakerEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-HighTorqueValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.high_torque_valve.env:AnymalHighTorqueValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-HighTorqueValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_hiveboard.tasks.anymal.high_torque_valve.env:AnymalHighTorqueValveEnvCfg_PLAY"
        ),
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-SmallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.small_valve.env:AnymalSmallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-SmallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.small_valve.env:AnymalSmallValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BenchValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.bench_valve.env:AnymalBenchValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BenchValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.bench_valve.env:AnymalBenchValveEnvCfg_PLAY",
    },
)
