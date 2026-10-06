# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm lock-and-key task.

The key is glued in the hand (see :mod:`isaaclab_hiveboard.tasks.scenes.key`):
the robot inserts it along the lock axis and turns the plug a quarter turn.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.events import set_contact_stiffness
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.key.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.anymal.key.configs.scene import KeySceneCfg
from isaaclab_hiveboard.tasks.anymal.key.configs.terminations import TerminationsCfg
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    MechanismPhysicsCfg,
    mechanism_events,
    mechanism_observations,
)
from isaaclab_hiveboard.tasks.viewer import use_play_viewer

KEY_JOINTS = ["RevoluteJoint"]


def key_contact_stiffness() -> EventTerm:
    """Stiff key/keyway contacts: MJWarp's mass-normalized default lets the 20 g plug sink under the blade.

    Same values as the small valve and the FR3 fingers. The key is a robot
    body, so its shapes are labelled ``.../Robot/.../Key/...``.
    """
    return EventTerm(
        func=set_contact_stiffness,
        mode="startup",
        params={"shape_regex": "/Key/|/Lock/", "ke": 4.0e4, "kd": 400.0, "solimp": (0.95, 0.99, 0.001)},
    )


def _key_events(play: bool = False) -> MechanismEventCfg:
    events = mechanism_events("lock", KEY_JOINTS, play=play)
    events.key_contacts = key_contact_stiffness()
    return events


@configclass
class AnymalKeyEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that inserts the glued key and turns it a quarter turn."""

    scene: KeySceneCfg = KeySceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = mechanism_observations("lock", KEY_JOINTS)  # type: ignore
    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: MechanismEventCfg = _key_events()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 15.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "lock"
        self.viewer.body_name = "lock"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.8, 0.8, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalKeyEnvCfg_PLAY(AnymalKeyEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = _key_events(play=True)  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        use_play_viewer(self)
