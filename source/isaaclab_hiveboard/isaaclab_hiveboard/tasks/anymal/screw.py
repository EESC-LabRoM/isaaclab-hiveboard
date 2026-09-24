# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP pieces for the HiveBoard screw tasks (M8 / M30 thread, peg).

The scripted expert regrasps the part and turns it up to
:data:`TURN_PER_GRASP_DEG` per grasp. The turn is the "valve" error to a goal
``turns`` revolutions tighter, so a slipped grasp is made up on the next one.
The command term couples the revolute to the prismatic joint at the thread
pitch, both as a per-step drive target and as a native Newton mimic
constraint (registered by :func:`configure_screw_env`).
"""

from __future__ import annotations

import math
from dataclasses import MISSING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
    ScrewJointCouplingCfg,
    SequentialPoseCommandCfg,
    register_screw_joint_mimic,
)
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_hiveboard.mdp.terminations import articulation_joint_ranges_success
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.actions import AnymalJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.mechanism import (
    ANYMAL_CUROBO,
    MechanismEventCfg,
    MechanismPhysicsCfg,
    mechanism_events,
)
from isaaclab_hiveboard.tasks.scenes.screw import ScrewSpec

SCREW_JOINTS = ["RevoluteJoint", "PrismaticJoint"]
# Hex nuts repeat every 60 deg, so the fixed grasp frame meets a flat again.
TURN_PER_GRASP_DEG = 120.0
# Seconds per regrasp cycle, for sizing the episode.
_CYCLE_S = 6.0


def _cycles(spec: ScrewSpec) -> int:
    # One spare grasp to make up slip.
    return math.ceil(spec.turns * 360.0 / TURN_PER_GRASP_DEG) + 1


def screw_episode_length_s(spec: ScrewSpec) -> float:
    return 10.0 + _CYCLE_S * _cycles(spec)


@configclass
class ScrewCommandsCfg:
    pose_command: SequentialPoseCommandCfg = MISSING  # type: ignore


def screw_commands(asset_name: str, spec: ScrewSpec) -> ScrewCommandsCfg:
    """Approach, then regrasp-and-turn cycles until seated, then back off."""

    def go(frame: str, velocity: float, threshold: float) -> CuroboPlannedGoToFrameCfg:
        return CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            gripper_open=True,
            distance_threshold=threshold,
            target_frame_name=frame,
            velocity=velocity,
            **ANYMAL_CUROBO,
        )

    cycle = [
        go("grasp", 0.1, 0.01),
        GripperCommand(open_gripper=False, duration_s=0.4),
        CuroboPlannedRotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="rotate_frame",
            # Tightening turns the part negative about object +X, which is -X
            # in the rotate frame (FACE_QUAT yaws it 180 deg).
            axis=(-1.0, 0.0, 0.0),
            max_ee_rotation_deg=TURN_PER_GRASP_DEG,
            angular_velocity=0.6,
            angle_threshold_deg=2.0,
            gripper_open=False,
            **ANYMAL_CUROBO,
        ),
        GripperCommand(open_gripper=True, duration_s=0.4),
        go("clear", 0.1, 0.01),
    ]
    tightened = -2.0 * math.pi * spec.turns
    return ScrewCommandsCfg(
        pose_command=SequentialPoseCommandCfg(
            asset_name="robot",
            body_name=ANYMAL_EE.body_name,
            resampling_time_range=(1e6, 1e6),
            debug_vis=False,
            output_joint_positions=True,
            valve_asset_name=asset_name,
            valve_joint_name="RevoluteJoint",
            open_task_prob=1.0,
            valve_joint_closed=0.0,
            valve_joint_open=tightened,
            valve_min_delta_rad=0.35,
            valve_ee_joint_angle_scale=1.0,
            commands=[go("approaching", 0.25, 0.03), *(cycle * _cycles(spec)), go("approaching", 0.2, 0.03)],
            body_offset=as_command_offset(ANYMAL_EE),
            screw_coupling=ScrewJointCouplingCfg(
                asset_name=asset_name,
                pitch_m_per_revolution=spec.pitch,
                lower_limit=0.0,
                upper_limit=spec.travel,
                # Thread friction is joint friction on the asset (see
                # screw_articulation); an explicit torque here blows up.
                viscous_friction=0.0,
                coulomb_friction=0.0,
                stiction=0.0,
                end_stop_base_damping=0.0,
                end_stop_scale=0.0,
                end_stop_activation_distance=0.0005,
                use_native_mimic_constraint=True,
            ),
        )
    )


def screw_events(asset_name: str, spec: ScrewSpec, play: bool = False) -> MechanismEventCfg:
    """Mechanism events with the thread friction randomized around ``spec.friction``."""
    events = mechanism_events(asset_name, ["RevoluteJoint"], play=play)
    if events.object_joint_parameters is not None:
        friction = (0.5 * spec.friction, 1.5 * spec.friction)
        events.object_joint_parameters.params["friction_distribution_params"] = friction
    return events


@configclass
class ScrewTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = MISSING  # type: ignore


def screw_terminations(asset_name: str, spec: ScrewSpec) -> ScrewTerminationsCfg:
    return ScrewTerminationsCfg(
        success=DoneTerm(
            func=articulation_joint_ranges_success,
            params={
                "command_name": "pose_command",
                "asset_name": asset_name,
                "ranges": {"PrismaticJoint": (-0.01, spec.seated_tolerance)},
            },
        )
    )


@configclass
class ScrewEnvCfg(ManagerBasedRLEnvCfg):
    """Base for the ANYmal screw tasks; subclasses fill the scene and MDP terms
    and call :func:`configure_screw_env` from ``__post_init__``."""

    actions: AnymalJointPositionActionCfg = AnymalJointPositionActionCfg()  # type: ignore
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=1, physics=MechanismPhysicsCfg())  # type: ignore
    rewards = None
    recorders: SpotManipulationRecorderCfg = SpotManipulationRecorderCfg()


def configure_screw_env(env_cfg: ScrewEnvCfg, asset_name: str, body_name: str, spec: ScrewSpec) -> None:
    """Episode length, viewer, recorders and the native screw constraint."""
    env_cfg.decimation = 15
    env_cfg.episode_length_s = screw_episode_length_s(spec)
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = asset_name
    env_cfg.viewer.body_name = body_name
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (-0.6, 0.6, 0.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # MODEL_INIT runs inside sim.reset(), before command terms exist.
    coupling = env_cfg.commands.pose_command.screw_coupling
    register_screw_joint_mimic(coupling, revolute_pos_at_reset=0.0, prismatic_pos_at_reset=spec.start)
