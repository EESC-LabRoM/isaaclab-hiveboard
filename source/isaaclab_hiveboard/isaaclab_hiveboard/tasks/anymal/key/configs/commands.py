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
# The hand turns this much more than the plug has left, to take up the
# keyway's ~5 deg of backlash; the plug's 90 deg stop takes the rest.
KEY_TURN_SCALE = 1.07


@configclass
class FramePoseCommandsCfg:
    """Carry the glued key to the lock, insert it along the lock axis and turn it a quarter turn.

    The key is welded to the hand, so the gripper only closes on its bow for
    show. The blade is not symmetric under a half turn, so no segment may
    canonicalize the TCP upright.
    """

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        # The plug is the "valve": the arc angle is its remaining error.
        valve_asset_name="lock",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=KEY_LOCKED,
        valve_joint_open=KEY_UNLOCKED,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=KEY_TURN_SCALE,
        commands=[
            GripperCommand(open_gripper=False, duration_s=0.4, phase="grip"),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.02,
                target_frame_name="key_approaching",
                velocity=0.25,
                phase="approach",
                canonicalize_upward=False,
                **ANYMAL_CUROBO,
            ),
            # Line the blade up with the keyway just off the face, then push
            # it in slowly: the slot leaves 1.2 mm per side.
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.003,
                orientation_threshold_deg=2.0,
                target_frame_name="key_entry",
                velocity=0.1,
                phase="engage",
                canonicalize_upward=False,
                **ANYMAL_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.003,
                orientation_threshold_deg=2.0,
                target_frame_name="key_inserted",
                velocity=0.03,
                phase="engage",
                canonicalize_upward=False,
                **ANYMAL_CUROBO,
            ),
            CuroboPlannedRotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # The key turns about object +X, which is -X in the rotate frame
                # (FACE_QUAT yaws it 180 deg).
                axis=(-1.0, 0.0, 0.0),
                angular_velocity=0.4,
                angle_threshold_deg=2.0,
                gripper_open=False,
                phase="actuate",
                **ANYMAL_CUROBO,
            ),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
