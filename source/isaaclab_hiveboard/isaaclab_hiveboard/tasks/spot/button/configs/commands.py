# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import SPOT_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Push the hinged cover open, then poke the button along its axis."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=SPOT_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                target_frame_name="approaching",
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                orientation_threshold_deg=35.0,
                velocity=0.12,
                target_frame_name="lid_contact",
            ),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="lid_hinge",
                axis=(0.0, 0.0, 1.0),
                angle_deg=60.0,
                angular_velocity=0.40,
                gripper_open=False,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.04,
                orientation_threshold_deg=40.0,
                velocity=0.20,
                target_frame_name="lid_clear",
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                velocity=0.20,
                target_frame_name="lid_safe",
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                orientation_threshold_deg=20.0,
                target_frame_name="button_approach",
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.015,
                orientation_threshold_deg=30.0,
                velocity=0.05,
                target_frame_name="button_press",
            ),
            GripperCommand(open_gripper=False, duration_s=0.6),
        ],
        body_offset=as_command_offset(SPOT_EE),
    )
