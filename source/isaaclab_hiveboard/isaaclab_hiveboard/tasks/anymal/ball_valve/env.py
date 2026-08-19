# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .configs.scene import AnymalBallValveSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class AnymalBallValveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ANYmal C HiveBoard Ball Valve environment."""

    scene: AnymalBallValveSceneCfg = AnymalBallValveSceneCfg(num_envs=1, env_spacing=3.0)
    observations: dict = {}
    actions: dict = {}
    events: dict = {}
    commands: dict = {}
    terminations: TerminationsCfg = TerminationsCfg()
    rewards = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        self.sim.dt = 1 / 200
        self.sim.render_interval = 10
