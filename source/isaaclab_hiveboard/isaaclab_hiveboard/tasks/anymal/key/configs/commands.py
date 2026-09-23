# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO

KEY_LOCKED = 0.0
KEY_UNLOCKED = math.pi / 2


@configclass
class FramePoseCommandsCfg:
    """Pinch the key bow, turn it a quarter turn, release and back off."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        # The key cylinder is the "valve": the arc angle is its remaining error.
        valve_asset_name="key",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=KEY_LOCKED,
        valve_joint_open=KEY_UNLOCKED,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="key_approaching",
                velocity=0.25,
                **ANYMAL_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.01,
                target_frame_name="key_grasp",
                velocity=0.1,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.4),
            CuroboPlannedRotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # The key turns about object +X, which is -X in the rotate frame
                # (FACE_QUAT yaws it 180 deg).
                axis=(-1.0, 0.0, 0.0),
                angular_velocity=0.4,
                angle_threshold_deg=2.0,
                gripper_open=False,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=True, duration_s=0.4),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="key_approaching",
                velocity=0.2,
                **ANYMAL_CUROBO,
            ),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
