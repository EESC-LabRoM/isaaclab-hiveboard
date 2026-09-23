# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm hidden-button task."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.button.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.anymal.button.configs.scene import ButtonSceneCfg
from isaaclab_hiveboard.tasks.anymal.button.configs.terminations import TerminationsCfg
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    MechanismPhysicsCfg,
    mechanism_events,
    mechanism_observations,
)

BUTTON_JOINTS = ["RevoluteJoint", "PrismaticJoint"]


@configclass
class AnymalButtonEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that opens the lid and presses the hidden button."""

    scene: ButtonSceneCfg = ButtonSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = mechanism_observations("button", BUTTON_JOINTS)  # type: ignore
    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    # Randomize the lid hinge friction; the button slide is a spring.
    events: MechanismEventCfg = mechanism_events("button", ["RevoluteJoint"])  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        # Two approaches, a ~120 deg lid arc at 0.4 rad/s, the press and holds.
        self.episode_length_s = 20.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "button"
        self.viewer.body_name = "button_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.8, 0.8, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalButtonEnvCfg_PLAY(AnymalButtonEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = mechanism_events("button", ["RevoluteJoint"], play=True)  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
