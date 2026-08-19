# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import (
    SpotIKAbsActionCfg,
    SpotIKRelativeActionCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.commands import (
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.events import (
    ValveEventCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.observations import (
    ObservationsCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.scene import (
    BallValveSceneCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.terminations import (
    DeltaCollectionTerminationsCfg,
    TerminationsCfg,
)
from isaaclab_hiveboard.tasks.spot.cabinet.configs.recorders import (
    SpotManipulationRecorderCfg,
)


@configclass
class SpotBallValveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ball valve environment."""

    scene: BallValveSceneCfg = BallValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotIKAbsActionCfg = SpotIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: ValveEventCfg = ValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        # general settings
        self.decimation = 5  #
        # Era 8
        self.episode_length_s = 8.0
        # Track the lever after reset. eye/lookat are offsets from that body.
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        # Front of the lever is +X in the valve body. Small Y/Z keep a 3/4 shot.
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 10
        # Prefer native-res AA over DLSS upscale for recorded video.
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.dlss_mode = 2  # Quality, if DLSS is selected instead
        self.sim.render.enable_reflections = True
        self.sim.render.enable_shadows = True
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 8
        self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)


@configclass
class SpotBallValveDeltaEnvCfg_PLAY(SpotBallValveEnvCfg):
    """Fixed-base ball-valve environment driven by relative TCP deltas."""

    actions: SpotIKRelativeActionCfg = SpotIKRelativeActionCfg()
    terminations: DeltaCollectionTerminationsCfg = DeltaCollectionTerminationsCfg()
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        super().__post_init__()
        # Collect and execute learned relative commands at 20 Hz. Command
        # phases are duration-based, so their wall-clock timing is unchanged.
        self.decimation = 10
        self.episode_length_s = 15.0
