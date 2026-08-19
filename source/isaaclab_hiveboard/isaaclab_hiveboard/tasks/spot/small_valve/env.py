# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import (
    SpotIKAbsActionCfg,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.observations import (
    ObservationsCfg,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.terminations import (
    TerminationsCfg,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.commands import (
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.events import (
    ValveEventCfg,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.scene import (
    SmallValveSceneCfg,
)


@configclass
class SpotSmallValveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the HiveBoard small gate valve environment."""

    scene: SmallValveSceneCfg = SmallValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotIKAbsActionCfg = SpotIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: ValveEventCfg = ValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 8.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "small_valve"
        self.viewer.body_name = "eixo_trans"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        self.sim.dt = 1 / 200
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)
