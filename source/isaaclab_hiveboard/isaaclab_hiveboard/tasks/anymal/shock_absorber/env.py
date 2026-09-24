# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab 3 configuration for the kitless ANYmal + DynaArm shock-absorber task."""

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset, make_ee_frame
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.mdp.terminations import articulation_joint_ranges_success
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    ANYMAL_CUROBO,
    MechanismEventCfg,
    MechanismObservationsCfg,
    MechanismPhysicsCfg,
    mechanism_events,
    mechanism_observations,
    pad_contacts,
)
from isaaclab_hiveboard.tasks.scenes.shock_absorber import PIN_SEATED, PIN_START, ShockAbsorberSceneCfg

PIN_JOINTS = ["PrismaticJoint"]
_LEFT_PAD, _RIGHT_PAD = pad_contacts("ShockAbsorber")


@configclass
class AnymalShockAbsorberSceneCfg(ShockAbsorberSceneCfg):
    """ANYmal + DynaArm + shared HiveBoard shock-absorber scene."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    finger_contact: ContactSensorCfg = _LEFT_PAD
    jaw_contact: ContactSensorCfg = _RIGHT_PAD


def _go(frame: str, velocity: float, threshold: float, gripper_open: bool = True, **kwargs):
    return CuroboPlannedGoToFrameCfg(
        frame_name="target_frame",
        gripper_open=gripper_open,
        distance_threshold=threshold,
        target_frame_name=frame,
        velocity=velocity,
        **kwargs,
        **ANYMAL_CUROBO,
    )


@configclass
class FramePoseCommandsCfg:
    """Grasp the pin head and push it in, then press it home with closed fingers."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        commands=[
            _go("pin_approaching", 0.25, 0.03),
            _go("pin_grasp", 0.1, 0.01),
            GripperCommand(open_gripper=False, duration_s=0.4),
            # The stop leaves the TCP short of the target by however far the
            # pin slipped; finish on the joint instead.
            _go(
                "pin_push",
                0.03,
                0.003,
                gripper_open=False,
                done_when_joint=("shock_absorber", "PrismaticJoint", -1.0, PIN_START - 0.010),
            ),
            GripperCommand(open_gripper=True, duration_s=0.4),
            _go("press_approaching", 0.1, 0.01),
            GripperCommand(open_gripper=False, duration_s=0.4),
            _go(
                "press",
                0.03,
                0.003,
                gripper_open=False,
                done_when_joint=("shock_absorber", "PrismaticJoint", -1.0, PIN_SEATED),
            ),
            _go("press_approaching", 0.1, 0.02, gripper_open=False),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=articulation_joint_ranges_success,
        params={
            "command_name": "pose_command",
            "asset_name": "shock_absorber",
            "ranges": {"PrismaticJoint": (-0.01, PIN_SEATED)},
        },
    )


@configclass
class AnymalShockAbsorberEnvCfg(ManagerBasedRLEnvCfg):
    """Randomized fixed-base ANYmal + DynaArm task that seats the shock-absorber pin."""

    scene: AnymalShockAbsorberSceneCfg = AnymalShockAbsorberSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: MechanismObservationsCfg = mechanism_observations("shock_absorber", PIN_JOINTS)  # type: ignore
    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: MechanismEventCfg = mechanism_events("shock_absorber", PIN_JOINTS)  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 30.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "shock_absorber"
        self.viewer.body_name = "pin"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.6, 0.6, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL


@configclass
class AnymalShockAbsorberEnvCfg_PLAY(AnymalShockAbsorberEnvCfg):
    """Deterministic one-environment Newton demonstration."""

    events: MechanismEventCfg = mechanism_events("shock_absorber", PIN_JOINTS, play=True)  # type: ignore
