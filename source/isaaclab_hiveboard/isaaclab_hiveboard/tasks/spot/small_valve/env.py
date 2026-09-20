# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless Spot small-valve task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import SpotJointPositionActionCfg
from isaaclab_hiveboard.tasks.spot.small_valve.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.spot.small_valve.configs.events import ValveEventCfg, ValvePlayEventCfg
from isaaclab_hiveboard.tasks.spot.small_valve.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.small_valve.configs.scene import SmallValveSceneCfg
from isaaclab_hiveboard.tasks.spot.small_valve.configs.terminations import TerminationsCfg


@configclass
class SpotSmallValvePhysicsCfg(PresetCfg):
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
        use_cuda_graph=False,
    )
    default = newton_mjwarp
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        friction_correlation_distance=0.00625,
    )


@configclass
class SpotSmallValveEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base Spot task that spins the small gate-valve handwheel."""

    scene: SmallValveSceneCfg = SmallValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotJointPositionActionCfg = SpotJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: ValveEventCfg = ValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=SpotSmallValvePhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 8.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "small_valve"
        self.viewer.body_name = "eixo_trans"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class SpotSmallValveEnvCfg_PLAY(SpotSmallValveEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: ValvePlayEventCfg = ValvePlayEventCfg()  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.commands.pose_command.open_task_prob = 1.0
