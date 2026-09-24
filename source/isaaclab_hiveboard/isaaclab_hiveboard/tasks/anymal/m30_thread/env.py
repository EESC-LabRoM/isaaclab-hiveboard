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


@configclass
class AnymalM30ThreadEnvCfg_PLAY(AnymalM30ThreadEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = screw_events("thread", M30_SPEC, play=True)  # type: ignore
