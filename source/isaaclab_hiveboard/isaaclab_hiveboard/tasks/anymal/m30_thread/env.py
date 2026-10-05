# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm M30 thread task."""

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, make_ee_frame
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import CuroboPlannedRotateFrameCfg, GripperCommand
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    MechanismEventCfg,
    MechanismObservationsCfg,
    mechanism_observations,
    pad_contacts,
)
from isaaclab_hiveboard.tasks.anymal.screw import (
    SCREW_JOINTS,
    ScrewCommandsCfg,
    ScrewEnvCfg,
    ScrewTerminationsCfg,
    configure_screw_env,
    screw_commands,
    screw_events,
    screw_terminations,
)
from isaaclab_hiveboard.tasks.scenes.threads import M30_SPEC, M30SceneCfg
from isaaclab_hiveboard.tasks.viewer import use_play_viewer

_LEFT_PAD, _RIGHT_PAD = pad_contacts("Thread")


@configclass
class AnymalM30ThreadSceneCfg(M30SceneCfg):
    """ANYmal + DynaArm + shared HiveBoard M30 thread scene."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    finger_contact: ContactSensorCfg = _LEFT_PAD
    jaw_contact: ContactSensorCfg = _RIGHT_PAD


@configclass
class AnymalM30ThreadEnvCfg(ScrewEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that runs the M30 nut down until seated."""

    scene: AnymalM30ThreadSceneCfg = AnymalM30ThreadSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = mechanism_observations("thread", SCREW_JOINTS)  # type: ignore
    terminations: ScrewTerminationsCfg = screw_terminations("thread", M30_SPEC)  # type: ignore
    events: MechanismEventCfg = screw_events("thread", M30_SPEC)  # type: ignore
    commands: ScrewCommandsCfg = screw_commands("thread", M30_SPEC)  # type: ignore

    def __post_init__(self):
        configure_screw_env(self, "thread", "Nut", M30_SPEC)
        # Allow the slower grasps and final retreat to finish before timeout.
        self.episode_length_s += 10.0
        # The native mimic constraint enforces the pitch at every substep.
        # A position drive anchored to the previous control step resists that
        # motion until the next target update, producing stick-slip at the pads.
        self.scene.thread.actuators["advance"].stiffness = 0.0
        self.scene.thread.actuators["advance"].damping = 0.0
        # Full closure (0.7 rad) crushes the 46 mm nut between the pads.
        # Ramp a modest squeeze and let it settle before starting the turn.
        self.actions.gripper_action.close_command_expr = {"finger_joint": 0.56}
        self.actions.gripper_action.close_speed = 0.7
        for command in self.commands.pose_command.commands:
            if isinstance(command, GripperCommand) and not command.open_gripper:
                command.duration_s = 0.9
            elif isinstance(command, CuroboPlannedRotateFrameCfg):
                command.screw_pitch_m_per_revolution = M30_SPEC.pitch


@configclass
class AnymalM30ThreadEnvCfg_PLAY(AnymalM30ThreadEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = screw_events("thread", M30_SPEC, play=True)  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        use_play_viewer(self)
