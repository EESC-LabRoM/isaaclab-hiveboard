# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO

# Drawer travel that counts as open (stop at 0.025 m).
DRAWER_OPEN = 0.02


@configclass
class FramePoseCommandsCfg:
    """Pinch the drawer front, pull it open, release and back off."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="drawer_approaching",
                velocity=0.25,
                **ANYMAL_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.01,
                target_frame_name="drawer_grasp",
                velocity=0.1,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.4),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.01,
                # A slipping grasp leaves the TCP ahead of the drawer; finish
                # on the drawer's own travel.
                done_when_joint=("drawer", "PrismaticJoint", DRAWER_OPEN, 1.0),
                target_frame_name="drawer_pulled",
                velocity=0.05,
                **ANYMAL_CUROBO,
            ),
            GripperCommand(open_gripper=True, duration_s=0.4),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="drawer_approaching",
                velocity=0.2,
                **ANYMAL_CUROBO,
            ),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
