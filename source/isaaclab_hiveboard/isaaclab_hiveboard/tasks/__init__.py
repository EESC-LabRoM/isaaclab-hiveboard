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
    id="Isaac-HiveBoard-Franka-BallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.ball_valve.env:FrankaBallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-BallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.ball_valve.env:FrankaBallValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-CircuitBreaker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.circuit_breaker.env:FrankaCircuitBreakerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-CircuitBreaker-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.circuit_breaker.env:FrankaCircuitBreakerEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-HighTorqueValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.high_torque_valve.env:FrankaHighTorqueValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-HighTorqueValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.high_torque_valve.env:FrankaHighTorqueValveEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-SmallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.small_valve.env:FrankaSmallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-SmallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.small_valve.env:FrankaSmallValveEnvCfg_PLAY",
    },
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
        "robomimic_bc_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.agents:robomimic/bc.json",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-BallValve-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.env:AnymalBallValveEnvCfg_PLAY",
        "robomimic_bc_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.agents:robomimic/bc.json",
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

##
# Hidden push button
##

gym.register(
    id="Isaac-HiveBoard-Anymal-Button-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.button.env:AnymalButtonEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-Button-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.button.env:AnymalButtonEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Button-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.button.env:FrankaButtonEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Button-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.button.env:FrankaButtonEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Button-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.button.env:SpotButtonEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Button-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.button.env:SpotButtonEnvCfg_PLAY",
    },
)

##
# Sliding drawer
##

gym.register(
    id="Isaac-HiveBoard-Spot-Drawer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.drawer.env:SpotDrawerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Drawer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.drawer.env:SpotDrawerEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-Drawer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.drawer.env:AnymalDrawerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-Drawer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.drawer.env:AnymalDrawerEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Drawer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.drawer.env:FrankaDrawerEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Drawer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.drawer.env:FrankaDrawerEnvCfg_PLAY",
    },
)

##
# Lock and key
##

gym.register(
    id="Isaac-HiveBoard-Spot-Key-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.key.env:SpotKeyEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Key-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.key.env:SpotKeyEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-Key-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.key.env:AnymalKeyEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-Key-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.key.env:AnymalKeyEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Key-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.key.env:FrankaKeyEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-Key-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.key.env:FrankaKeyEnvCfg_PLAY",
    },
)

##
# M8 thread
##

gym.register(
    id="Isaac-HiveBoard-Spot-M8Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.m8_thread.env:SpotM8ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-M8Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.m8_thread.env:SpotM8ThreadEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-M8Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.m8_thread.env:AnymalM8ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-M8Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.m8_thread.env:AnymalM8ThreadEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-M8Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.m8_thread.env:FrankaM8ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-M8Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.m8_thread.env:FrankaM8ThreadEnvCfg_PLAY",
    },
)

##
# M30 thread
##

gym.register(
    id="Isaac-HiveBoard-Spot-M30Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.m30_thread.env:SpotM30ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-M30Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.m30_thread.env:SpotM30ThreadEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-M30Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.m30_thread.env:AnymalM30ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-M30Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.m30_thread.env:AnymalM30ThreadEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-M30Thread-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.m30_thread.env:FrankaM30ThreadEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-M30Thread-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.m30_thread.env:FrankaM30ThreadEnvCfg_PLAY",
    },
)

##
# Peg insertion
##

gym.register(
    id="Isaac-HiveBoard-Spot-PegInsertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.peg_insertion.env:SpotPegInsertionEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-PegInsertion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.peg_insertion.env:SpotPegInsertionEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-PegInsertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.peg_insertion.env:AnymalPegInsertionEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-PegInsertion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.peg_insertion.env:AnymalPegInsertionEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-PegInsertion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.peg_insertion.env:FrankaPegInsertionEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-PegInsertion-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.peg_insertion.env:FrankaPegInsertionEnvCfg_PLAY",
    },
)

##
# Shock absorber
##

gym.register(
    id="Isaac-HiveBoard-Spot-ShockAbsorber-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.shock_absorber.env:SpotShockAbsorberEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-ShockAbsorber-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.shock_absorber.env:SpotShockAbsorberEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-ShockAbsorber-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.shock_absorber.env:AnymalShockAbsorberEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-ShockAbsorber-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.shock_absorber.env:AnymalShockAbsorberEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-ShockAbsorber-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.shock_absorber.env:FrankaShockAbsorberEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Franka-ShockAbsorber-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.shock_absorber.env:FrankaShockAbsorberEnvCfg_PLAY",
    },
)
