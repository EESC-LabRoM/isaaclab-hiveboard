# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless Spot ball-valve task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import SpotIKAbsActionCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.events import ValveEventCfg, ValvePlayEventCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.scene import BallValveSceneCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.terminations import DeltaCollectionTerminationsCfg


@configclass
class SpotBallValvePhysicsCfg(PresetCfg):
    """Physics variants exposed through ``physics=...`` on the Isaac Lab 3 CLI."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=4000,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        friction_correlation_distance=0.00625,
    )


@configclass
class SpotBallValveEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base Spot task that physically opens the valve to -90 degrees."""

    scene: BallValveSceneCfg = BallValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotIKAbsActionCfg = SpotIKAbsActionCfg()  # type: ignore
    terminations: DeltaCollectionTerminationsCfg = DeltaCollectionTerminationsCfg()  # type: ignore
    events: ValveEventCfg = ValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 600, render_interval=30, physics=SpotBallValvePhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        # 600 Hz physics / 30 = 20 Hz command and action rate.
        self.decimation = 30
        self.episode_length_s = 10.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # Keep failed episodes so contact traces are available for debugging.
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class SpotBallValveEnvCfg_PLAY(SpotBallValveEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: ValvePlayEventCfg = ValvePlayEventCfg()  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.pose_command.open_task_prob = 1.0
