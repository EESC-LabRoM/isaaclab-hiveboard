# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Approach the lever, close the 2F-140, and rotate the valve open."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        # HiveBoard limits are [-pi/2, 0]: negative rotation opens the valve.
        valve_joint_closed=0.0,
        valve_joint_open=-1.5707963267948966,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                # Position-only DLS cannot track orientation; don't block the sequence on it.
                orientation_threshold_deg=180.0,
                canonicalize_upward=False,
                target_frame_name="approaching",
                velocity=0.25,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                # Close while seating on the lever so the 2F-140 shuts even if
                # position-IK never quite hits the GripperCommand threshold.
                gripper_open=False,
                distance_threshold=0.08,
                orientation_threshold_deg=180.0,
                canonicalize_upward=False,
                target_frame_name="lever_pivot",
                velocity=0.15,
            ),
            GripperCommand(open_gripper=False, duration_s=1.25),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # Fallback only; the command term uses the remaining valve error.
                angle_deg=-90,
                angular_velocity=0.3,
                gripper_open=False,
            ),
            GripperCommand(open_gripper=False, duration_s=0.075),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
