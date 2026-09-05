# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import (
    SpotIKAbsActionCfg,
)
from isaaclab_hiveboard.tasks.spot.button.configs.commands import (
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.spot.button.configs.events import (
    ButtonEventCfg,
)
from isaaclab_hiveboard.tasks.spot.button.configs.observations import (
    ObservationsCfg,
)
from isaaclab_hiveboard.tasks.spot.button.configs.scene import (
    ButtonSceneCfg,
)
from isaaclab_hiveboard.tasks.spot.button.configs.terminations import (
    TerminationsCfg,
)


@configclass
class SpotButtonEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the HiveBoard hidden-button environment."""

    scene: ButtonSceneCfg = ButtonSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotIKAbsActionCfg = SpotIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: ButtonEventCfg = ButtonEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 12.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "button"
        self.viewer.body_name = "lid_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.45, 0.35, 0.12)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)
