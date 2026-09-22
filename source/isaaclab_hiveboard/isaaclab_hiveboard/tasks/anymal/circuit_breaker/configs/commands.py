# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR, ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)

ANYMAL_ARM_6 = list(ANYMAL_ARM_JOINT_NAMES)
_CUROBO = {
    "robot_joint_names": ANYMAL_ARM_6,
    "robot_curobo_yaml": f"{ASSET_DIR}/anymal/cumotion/dynaarm.yaml",
    "robot_urdf": f"{ASSET_DIR}/anymal/cumotion/dynaarm.urdf",
}


@configclass
class FramePoseCommandsCfg:
    """Approach the circuit breaker lever, flip the switch, then retreat."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        valve_asset_name="circuit_breaker",
        valve_joint_name="RevoluteJoint",
        open_task_prob=0.0,
        # HiveBoard limits are [-pi/6, +pi/6].
        valve_joint_closed=0.5235987755982988,
        valve_joint_open=-0.5235987755982988,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                target_frame_name="approaching",
                velocity=0.25,
                **_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.02,
                target_frame_name="lever_pivot_below",
                velocity=0.15,
                **_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.3),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.02,
                target_frame_name="lever_pivot_above",
                velocity=0.15,
                **_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.375),
            GripperCommand(open_gripper=False, duration_s=0.375),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                target_frame_name="approaching",
                velocity=0.25,
                **_CUROBO,
            ),
            GripperCommand(open_gripper=True, duration_s=0.75),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
