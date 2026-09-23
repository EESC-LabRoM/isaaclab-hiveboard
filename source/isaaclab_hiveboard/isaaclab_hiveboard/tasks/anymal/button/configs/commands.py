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
from isaaclab_hiveboard.tasks.scenes.button import LID_CLOSED

# Past the lid's upright pose (0) toward its 40 deg stop, so the open lid
# stands clear of the button face.
LID_OPEN = math.radians(30.0)
# Button slide position that counts as pressed (stop at -0.010).
BUTTON_PRESSED = -0.008


@configclass
class FramePoseCommandsCfg:
    """Pinch the lid, swing it open about its hinge, then press the button."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        # The lid hinge is the "valve": the arc angle is the lid's remaining error.
        valve_asset_name="button",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=LID_CLOSED,
        valve_joint_open=LID_OPEN,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="lid_approaching",
                velocity=0.25,
                **ANYMAL_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.01,
                target_frame_name="lid_grasp",
                velocity=0.1,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.4),
            CuroboPlannedRotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # Hinge axis is object +Z; the rotate frame only yaws, so +Z holds.
                axis=(0.0, 0.0, 1.0),
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
                target_frame_name="button_approaching",
                velocity=0.2,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.4),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.015,
                # The stop holds the TCP short of the target by however far the
                # closed fingers reach past it; finish on the button instead.
                done_when_joint=("button", "PrismaticJoint", -1.0, BUTTON_PRESSED),
                target_frame_name="button_press",
                velocity=0.05,
                **ANYMAL_CUROBO,
            ),
            # Keep pushing while the success term reads the pressed button.
            GripperCommand(open_gripper=False, duration_s=0.5),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
