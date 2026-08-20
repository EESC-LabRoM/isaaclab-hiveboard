# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import (
    SpotIKAbsActionCfg,
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
    TerminationsCfg,
)
from isaaclab_hiveboard.mdp.recorders import (
    SpotManipulationCameraRecorderCfg,
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
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()
    rewards = None

    def __post_init__(self):
        # general settings
        self.decimation = 10  #
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
        # Prefer native-res AA over DLSS upscale for recorded video.
        self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.init_state.pos = (0, 0, 0)


@configclass
class SpotBallValveEnvCfg_PLAY(SpotBallValveEnvCfg):
    """Fixed-base ball-valve environment driven by relative TCP deltas."""

    def __post_init__(self):
        super().__post_init__()
        # Collect and execute learned relative commands at 20 Hz. Command
        # phases use constant velocity, so wall-clock timing scales with distance.
        self.decimation = 10  # 20 Hz


def _add_collection_cameras(cfg: ManagerBasedRLEnvCfg) -> None:
    """Wrist and third-person RGB cameras stored in the HDF5 dataset."""
    cfg.scene.wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1/wrist_cam",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.18, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )
    cfg.scene.scene_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.5, 1.5, 0.6),
            rot=(-0.35355, 0.61237, 0.61237, -0.35355),
            convention="ros",
        ),
    )
    cfg.num_rerenders_on_reset = 3


@configclass
class SpotBallValveEnvCfg_PLAY_CAMERAS(SpotBallValveEnvCfg_PLAY):
    """PLAY collection env with wrist and scene RGB cameras."""

    recorders: SpotManipulationCameraRecorderCfg = SpotManipulationCameraRecorderCfg()

    def __post_init__(self):
        super().__post_init__()
        _add_collection_cameras(self)
