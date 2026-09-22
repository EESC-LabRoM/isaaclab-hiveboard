# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.anymal.bench import (
    FLANGE_TO_TCP_POS,
    FLANGE_TO_TCP_QUAT_XYZW,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
)
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)

from isaaclab_hiveboard.utils.spot_traj import BenchKeyHold, BenchKeyMove, bench_key_steps


def _key_commands():
    """Visit the same traj_edit beads as the Spot bench env, including gripper holds.

    Targets come from :func:`bench_key_steps` (TCP poses in the env frame; the
    board/valve placement is shared with Spot's bench), each solved for the
    DynaArm flange through the flange->TCP offset.
    """
    commands = []
    for step in bench_key_steps():
        if isinstance(step, BenchKeyHold):
            commands.append(GripperCommand(open_gripper=step.gripper_open, duration_s=step.duration_s))
            continue
        commands.append(
            GoToFrameCfg(
                frame_name="",
                target_frame_name="",
                target_position_env=step.pos,
                target_orientation_env=step.quat_xyzw,
                gripper_open=step.gripper_open,
                velocity=0.15,
                distance_threshold=0.01,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
                valve_done_threshold_rad=0.10
                if isinstance(step, BenchKeyMove) and step.complete_on_valve
                else None,
            )
        )
    print(f"[AnymalBenchValveCommands] final command ({len(commands)} steps):")
    for i, cmd in enumerate(commands):
        print(f"  [{i}] {cmd}")
    return commands


@configclass
class BenchValveCommandsCfg:
    """Sequential Cartesian targets from the trajectory editor's key beads."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="dynaarm_flange",
        body_offset=SequentialPoseCommandCfg.OffsetCfg(pos=FLANGE_TO_TCP_POS, rot=FLANGE_TO_TCP_QUAT_XYZW),
        resampling_time_range=(1e6, 1e6),
        commands=_key_commands(),
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=VALVE_JOINT_CLOSED,
        valve_joint_open=VALVE_JOINT_OPEN,
        debug_vis=True,
    )
