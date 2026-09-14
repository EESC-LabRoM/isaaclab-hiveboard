# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.anymal.bench import (
    FLANGE_TO_TCP_POS,
    FLANGE_TO_TCP_QUAT_XYZW,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
)
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.anymal.curobo_valve.configs.actions import ANYMAL_ARM_6

from isaaclab_hiveboard.utils.spot_traj import (
    BenchKeyHold,
    BenchKeyMove,
    bench_arc_reference,
    bench_key_steps,
)

ANYMAL_CUROBO_YAML = f"{ASSET_DIR}/anymal/cumotion/dynaarm.yaml"
ANYMAL_URDF = f"{ASSET_DIR}/anymal/cumotion/dynaarm.urdf"


def _key_commands():
    """Same traj_edit beads as the Spot bench/curobo valves, cuRobo-planned.

    Targets are TCP poses in the env frame (board/valve placement is shared
    with the Spot bench), so the Cartesian goals are identical. Each leg is
    solved for the DynaArm flange; the flange→TCP offset converts them.
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
                robot_joint_names=list(ANYMAL_ARM_6),
                robot_curobo_yaml=ANYMAL_CUROBO_YAML,
                robot_urdf=ANYMAL_URDF,
                num_ik_seeds=16,
                num_trajopt_seeds=2,
                max_plan_attempts=3,
            )
        )
    print(f"[AnymalCuroboValveCommands] final command ({len(commands)} steps):")
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
class AnymalCuroboValveCommandsCfg:
    """Sequential Cartesian targets from traj_edit, executed via cuRobo plans."""

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
        log_transitions=True,
        stall_timeout_s=8.0,
        path_debug_vis=True,
    )
