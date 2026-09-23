# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm sliding-drawer task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.scene import DrawerSceneCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.terminations import TerminationsCfg
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    MechanismPhysicsCfg,
    mechanism_events,
    mechanism_observations,
)

DRAWER_JOINTS = ["PrismaticJoint"]


@configclass
class AnymalDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that pulls the drawer open."""

    scene: DrawerSceneCfg = DrawerSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = mechanism_observations("drawer", DRAWER_JOINTS)  # type: ignore
    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: MechanismEventCfg = mechanism_events("drawer", DRAWER_JOINTS)  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 12.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "drawer"
        self.viewer.body_name = "drawer"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.8, 0.8, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalDrawerEnvCfg_PLAY(AnymalDrawerEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = mechanism_events("drawer", DRAWER_JOINTS, play=True)  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
