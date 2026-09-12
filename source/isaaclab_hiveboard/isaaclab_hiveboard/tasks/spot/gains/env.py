# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-only Spot env for joint-space PD gain tuning."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.assets.spot.bench import DECIMATION, PHYSICS_DT, TRAJ_RATE_HZ
from isaaclab_hiveboard.tasks.spot.gains.configs.actions import SpotGainsActionCfg
from isaaclab_hiveboard.tasks.spot.gains.configs.commands import SpotGainsCommandsCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.events import BenchValveEventCfg
from isaaclab_hiveboard.tasks.spot.gains.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.gains.configs.scene import SpotGainsSceneCfg
from isaaclab_hiveboard.tasks.spot.gains.configs.terminations import SpotGainsTerminationsCfg


@configclass
class SpotGainsPhysicsCfg(PresetCfg):
    """Physics variants. CUDA graph is off so live gain writes take effect."""

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
class SpotGainsEnvCfg(ManagerBasedRLEnvCfg):
    """Replay the website arm clip in free space (no HiveBoard)."""

    scene: SpotGainsSceneCfg = SpotGainsSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotGainsActionCfg = SpotGainsActionCfg()  # type: ignore
    terminations: SpotGainsTerminationsCfg = SpotGainsTerminationsCfg()  # type: ignore
    events: BenchValveEventCfg = BenchValveEventCfg()  # type: ignore
    commands: SpotGainsCommandsCfg = SpotGainsCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics=SpotGainsPhysicsCfg(),
    )  # type: ignore
    rewards = None
    recorders = None

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 636 / TRAJ_RATE_HZ + 0.28
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.8, -2.0, 1.2)
        self.viewer.lookat = (0.4, 0.0, 0.6)


@configclass
class SpotGainsEnvCfg_PLAY(SpotGainsEnvCfg):
    """Deterministic one-environment gain-tuning play cfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
