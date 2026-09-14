# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal bench valve executed through cuRobo motion plans."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.anymal.bench import DECIMATION, PHYSICS_DT
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.actions import (
    AnymalCuroboValveActionCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.commands import (
    AnymalCuroboValveCommandsCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.events import (
    AnymalCuroboValveEventCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.observations import (
    AnymalCuroboObservationsCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.scene import (
    AnymalBenchValveSceneCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.terminations import (
    AnymalCuroboValveTerminationsCfg,
)
from isaaclab_hiveboard.tasks.spot.bench_valve.env import SpotBenchValvePhysicsCfg


@configclass
class AnymalCuroboValveEnvCfg(ManagerBasedRLEnvCfg):
    """Bench valve keys with each Cartesian leg planned by cuRobo for the DynaArm."""

    scene: AnymalBenchValveSceneCfg = AnymalBenchValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: AnymalCuroboObservationsCfg = AnymalCuroboObservationsCfg()  # type: ignore
    actions: AnymalCuroboValveActionCfg = AnymalCuroboValveActionCfg()  # type: ignore
    terminations: AnymalCuroboValveTerminationsCfg = AnymalCuroboValveTerminationsCfg()  # type: ignore
    events: AnymalCuroboValveEventCfg = AnymalCuroboValveEventCfg()  # type: ignore
    commands: AnymalCuroboValveCommandsCfg = AnymalCuroboValveCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics=SpotBenchValvePhysicsCfg(),
    )  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 30.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (0.2, -2.1, 0.9)
        self.viewer.lookat = (0.7, 0.0, 0.7)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalCuroboValveEnvCfg_PLAY(AnymalCuroboValveEnvCfg):
    """Deterministic one-environment cuRobo demonstration."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
