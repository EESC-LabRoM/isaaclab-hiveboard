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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_slide_observation
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    MechanismPhysicsCfg,
    mechanism_events,
    mechanism_observations,
)


def _drawer_observations() -> MechanismObservationsCfg:
    """Shared mechanism observations, with the slide distance in place of a joint."""
    observations = mechanism_observations("drawer", [])
    params = {
        "box_cfg": SceneEntityCfg("drawer"),
        "housing_cfg": SceneEntityCfg("drawer_housing"),
    }
    observations.diffusion_policy.object_joint_pos = ObsTerm(func=drawer_slide_observation, params=dict(params))
    observations.evaluation.object_joint_pos = ObsTerm(func=drawer_slide_observation, params=dict(params))
    return observations


def _drawer_events(play: bool = False) -> MechanismEventCfg:
    """Events for a free box. There is no joint to randomize, and gravity is already off."""
    events = mechanism_events("drawer", [], play=play)
    events.object_joint_parameters = None
    events.object_gravcomp = None
    return events


@configclass
class AnymalDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that pulls the drawer open."""

    scene: DrawerSceneCfg = DrawerSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = _drawer_observations()  # type: ignore
    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: MechanismEventCfg = _drawer_events()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 12.0
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "drawer_housing"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.8, 0.8, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalDrawerEnvCfg_PLAY(AnymalDrawerEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = _drawer_events(play=True)  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
