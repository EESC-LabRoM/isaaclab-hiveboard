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

