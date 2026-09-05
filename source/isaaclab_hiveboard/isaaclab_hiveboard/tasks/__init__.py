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
    id="Isaac-HiveBoard-Spot-BallValve-Play-Cameras-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_hiveboard.tasks.spot.ball_valve.env:SpotBallValveEnvCfg_PLAY_CAMERAS"
        ),
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
    id="Isaac-HiveBoard-Spot-CircuitBreaker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.circuit_breaker.env:SpotCircuitBreakerEnvCfg",
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
    id="Isaac-HiveBoard-Spot-SmallValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.small_valve.env:SpotSmallValveEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Spot-Lamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.lamp.env:SpotLampEnvCfg",
    },
)

##
# Franka Environments
##

gym.register(
    id="Isaac-HiveBoard-Franka-Lamp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.lamp.env:FrankaLampEnvCfg"},
)

gym.register(
    id="Isaac-HiveBoard-Franka-LeverValve-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.lever_valve.env:FrankaLeverValveEnvCfg",
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
    id="Isaac-HiveBoard-Franka-OnlyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.only_robot.env:FrankaOnlyRobotEnvCfg",
    },
)

##
# ANYmal Environments
##

gym.register(
    id="Isaac-HiveBoard-Anymal-OnlyRobot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.only_robot.env:AnymalOnlyRobotEnvCfg",
    },
)

gym.register(
    id="Isaac-HiveBoard-Anymal-OnlyGripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.only_gripper.env:AnymalOnlyGripperEnvCfg",
    },
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
    id="Anymal-Manipulation-Ball-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.anymal.ball_valve.env:AnymalBallValveEnvCfg",
    },
)

##
# Backwards-compatibility aliases
##

gym.register(
    id="Spot-Manipulation-Ball-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.ball_valve.env:SpotBallValveEnvCfg",
    },
)

gym.register(
    id="Spot-Manipulation-Button",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.button.env:SpotButtonEnvCfg",
    },
)

gym.register(
    id="Spot-Manipulation-Circuit-Breaker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.circuit_breaker.env:SpotCircuitBreakerEnvCfg",
    },
)

gym.register(
    id="Spot-Manipulation-High-Torque-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.high_torque_valve.env:SpotHighTorqueValveEnvCfg",
    },
)

gym.register(
    id="Spot-Manipulation-Small-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.small_valve.env:SpotSmallValveEnvCfg",
    },
)

gym.register(
    id="Spot-Manipulation-Lamp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.spot.lamp.env:SpotLampEnvCfg",
    },
)

gym.register(
    id="Franka-Manipulation-Lever-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.lever_valve.env:FrankaLeverValveEnvCfg",
    },
)

gym.register(
    id="Franka-Manipulation-Ball-Valve",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.lever_valve.env:FrankaLeverValveEnvCfg",
    },
)

gym.register(
    id="Franka-Manipulation-Circuit-Breaker",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_hiveboard.tasks.franka.circuit_breaker.env:FrankaCircuitBreakerEnvCfg",
    },
)
