# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm sliding-drawer task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import AnymalDrawerCommandsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.scene import DrawerSceneCfg
from isaaclab_hiveboard.tasks.anymal.drawer.configs.terminations import DrawerRemovalTerminationsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.physics import DRAWER_DECIMATION, DRAWER_SIM_DT, DrawerPhysicsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_slide_observation
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    mechanism_events,
    mechanism_observations,
)
from isaaclab_hiveboard.tasks.viewer import use_play_viewer


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
    """Free drawer with gravity and separate guide/handle contact materials."""
    events = mechanism_events("drawer", [], play=play)
    events.object_joint_parameters = None
    events.object_gravcomp = None
    # The USD assigns low sliding friction to shafts/walls and higher pad
    # friction to the handle. A body-wide material randomizer erases that split.
    events.object_physics_material = None
    return events


def _drawer_actions() -> AnymalJointPositionActionCfg:
    actions = AnymalJointPositionActionCfg()
    # Fully spread fingers hit the side cuts. This opening clears the 30 mm
    # handle tab while fitting between the rails; the close target stays 0.7.
    actions.gripper_action.open_command_expr = {"finger_joint": 0.61}
    actions.gripper_action.close_speed = 0.2
    actions.gripper_action.open_speed = 0.8
    return actions


@configclass
class AnymalDrawerEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that extracts the drawer."""

    scene: DrawerSceneCfg = DrawerSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = _drawer_observations()  # type: ignore
    actions: AnymalJointPositionActionCfg = _drawer_actions()  # type: ignore
    terminations: DrawerRemovalTerminationsCfg = DrawerRemovalTerminationsCfg()  # type: ignore
    events: MechanismEventCfg = _drawer_events()  # type: ignore
    commands: AnymalDrawerCommandsCfg = AnymalDrawerCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=DRAWER_SIM_DT, render_interval=1, physics=DrawerPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = DRAWER_DECIMATION
        self.episode_length_s = 20.0
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
        use_play_viewer(self)
