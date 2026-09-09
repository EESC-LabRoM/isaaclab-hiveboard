# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 env that replays the HiveBoard website Spot valve clip."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.assets.spot.bench import DECIMATION, PHYSICS_DT, TRAJ_RATE_HZ
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.actions import SpotBenchJointActionCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.commands import BenchValveCommandsCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.events import BenchValveEventCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.scene import BenchValveSceneCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.terminations import BenchValveTerminationsCfg


@configclass
class SpotBenchValvePhysicsCfg(PresetCfg):
    """Physics variants exposed through ``physics=...`` on the Isaac Lab 3 CLI."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=400,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    default = newton_mjwarp
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        friction_correlation_distance=0.00625,
    )


@configclass
class SpotBenchValveEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-trajectory Spot valve playback matching the website demo."""

    scene: BenchValveSceneCfg = BenchValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotBenchJointActionCfg = SpotBenchJointActionCfg()  # type: ignore
    terminations: BenchValveTerminationsCfg = BenchValveTerminationsCfg()  # type: ignore
    events: BenchValveEventCfg = BenchValveEventCfg()  # type: ignore
    commands: BenchValveCommandsCfg = BenchValveCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics=SpotBenchValvePhysicsCfg(),
    )  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        # 500 Hz physics / 10 = 50 Hz, matching the website clip.
        self.decimation = DECIMATION
        # 636 samples at 50 Hz = 12.72 s, plus a beat on the stop.
        self.episode_length_s = 636 / TRAJ_RATE_HZ + 0.28
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.2, -2.1, 0.9)
        self.viewer.lookat = (0.7, 0.0, 0.7)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class SpotBenchValveEnvCfg_PLAY(SpotBenchValveEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
