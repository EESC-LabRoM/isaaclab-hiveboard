# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.cabinet.configs.actions import (
    RMPActionsCfg,
    SpotIKRelativeActionCfg,
)
from isaaclab_hiveboard.tasks.spot.cabinet.configs.commands import (
    CuroboCommandsCfg,
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.spot.cabinet.configs.events import EventCfg
from isaaclab_hiveboard.tasks.spot.cabinet.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.cabinet.configs.scene import CabinetSceneCfg
from isaaclab_hiveboard.tasks.spot.cabinet.configs.terminations import TerminationsCfg


@configclass
class SpotRMPCabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=1, env_spacing=3.0)  # from 4096
    observations: ObservationsCfg = ObservationsCfg()
    actions: RMPActionsCfg = RMPActionsCfg()
    events: EventCfg = EventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards = None

    def __post_init__(self):
        # general settings
        self.decimation = 5  #
        # Era 8
        self.episode_length_s = 30.0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.3, 0.0, 0)
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 10
        self.sim.render.antialiasing_mode = "DLSS"
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        # The stiffness is set to the value parsed from the URDF file
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)


@configclass
class SpotCuroboCabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=1, env_spacing=3.0)  # from 4096
    observations: ObservationsCfg = ObservationsCfg()
    actions: SpotIKRelativeActionCfg = SpotIKRelativeActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands: CuroboCommandsCfg = CuroboCommandsCfg()
    rewards = None

    def __post_init__(self):
        # general settings
        self.decimation = 5  #
        # Era 8
        self.episode_length_s = 30.0
        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.3, 0.0, 0)
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 10
        self.sim.render.antialiasing_mode = "DLSS"
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)
