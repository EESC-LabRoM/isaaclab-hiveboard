# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.spot.bench import (
    TCP_SITE_POS,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
)
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.spot.curobo_valve.configs.actions import SPOT_ARM_6

from isaaclab_hiveboard.utils.spot_traj import (
    BenchKeyHold,
    BenchKeyMove,
    bench_arc_reference,
    bench_key_steps,
)

SPOT_CUROBO_YAML = f"{ASSET_DIR}/spot/cumotion/spot_arm.yaml"
SPOT_URDF = f"{ASSET_DIR}/spot/spot_with_arm.urdf"


def _key_commands():
    """Same traj_edit beads as bench_valve, each leg cuRobo-planned.

    Targets come from :func:`bench_key_steps`, shared with the bench env so
    both track identical Cartesian goals.
    """
    commands = []
    arc_reference = bench_arc_reference()
    for step in bench_key_steps():
        if isinstance(step, BenchKeyHold):
            commands.append(GripperCommand(open_gripper=step.gripper_open, duration_s=step.duration_s))
            continue
        use_reference = (
            isinstance(step, BenchKeyMove) and step.complete_on_valve and arc_reference is not None
        )
        commands.append(
            CuroboPlannedGoToFrameCfg(
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
                reference_pos_env=tuple(arc_reference[0]) if use_reference else None,
                reference_quat_xyzw=tuple(arc_reference[1]) if use_reference else None,
                robot_joint_names=list(SPOT_ARM_6),
                robot_curobo_yaml=SPOT_CUROBO_YAML,
                robot_urdf=SPOT_URDF,
                num_ik_seeds=16,
                num_trajopt_seeds=2,
                max_plan_attempts=3,
            )
        )
    print(f"[CuroboValveCommands] final command ({len(commands)} steps):")
    for i, cmd in enumerate(commands):
        if isinstance(cmd, CuroboPlannedGoToFrameCfg) and cmd.reference_pos_env is not None:
            print(
                f"  [{i}] CuroboPlannedGoToFrameCfg(target_position_env={cmd.target_position_env}, "
                f"gripper_open={cmd.gripper_open}, "
                f"cartesian_reference={len(cmd.reference_pos_env)} samples)"
            )
        else:
            print(f"  [{i}] {cmd}")
    return commands


@configclass
class CuroboValveCommandsCfg:
    """Sequential Cartesian targets from traj_edit, executed via cuRobo plans."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_wr1",
        body_offset=SequentialPoseCommandCfg.OffsetCfg(pos=TCP_SITE_POS),
        resampling_time_range=(1e6, 1e6),
        commands=_key_commands(),
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=VALVE_JOINT_CLOSED,
        valve_joint_open=VALVE_JOINT_OPEN,
        debug_vis=True,
        log_transitions=True,
        stall_timeout_s=8.0,
        path_debug_vis=True,
    )
