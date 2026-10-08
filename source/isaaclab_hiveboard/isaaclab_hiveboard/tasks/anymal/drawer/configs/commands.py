# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO

# Forward travel that counts as open for the shared open-only sequence.
# ANYmal additionally requires the lift and complete removal from the case.
DRAWER_OPEN = 0.02

# The drawer needs contact near the pad tips: the shared 200 mm mid-pad TCP
# inserts the fingers too far into its shallow front. Move the reference
# 15 mm forward without changing the jaw axes or other ANYmal tasks.
ANYMAL_DRAWER_EE = ANYMAL_EE.replace(
    tcp_offset=ANYMAL_EE.tcp_offset.replace(pos=(0.0, 0.0, 0.215)),
)


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
                # on the box's forward travel in the housing frame.
                done_when_travel=("drawer", "drawer_housing", DRAWER_OPEN, 1.0),
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


@configclass
class AnymalDrawerCommandsCfg(FramePoseCommandsCfg):
    """Pinch, pull to the stop, lift through the side cuts, and remove the drawer."""

    def __post_init__(self):
        self.pose_command.body_offset = as_command_offset(ANYMAL_DRAWER_EE)
        # Pitch the TCP down 25 degrees to reach the handle from above the
        # guides, keeping its +Y jaws across the tab's 30 mm width.
        half_pitch = math.radians(25.0) / 2.0
        pitch = (0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch))
        for command in self.pose_command.commands:
            if isinstance(command, CuroboPlannedGoToFrameCfg):
                command.target_offset_rot = pitch
                command.canonicalize_upward = False
                command.distance_threshold = 0.003
        # Enter and leave above the handle; descend only after the wrist is
        # aligned. The pull remains level so the shafts stay in the side slots.
        self.pose_command.commands[0].target_offset_pos = (0.0, 0.0, 0.04)
        self.pose_command.commands[-1].target_offset_pos = (0.0, 0.0, 0.04)
        self.pose_command.commands[2].duration_s = 1.0
        add_drawer_removal_steps(self.pose_command)


def add_drawer_removal_steps(pose_command: SequentialPoseCommandCfg, exit_height: float = 0.0125) -> None:
    """Extend the six-step opening sequence with a contact-guided extraction.

    Clone the robot's pull command so its TCP orientation and planner settings
    carry through the lift. Offsets use FACE_QUAT: negative X pulls outward,
    positive Z lifts in the housing frame.
    """
    commands = pose_command.commands
    pull = commands[3]
    pull.phase = "actuate"
    pull.target_offset_pos = (-0.0026, 0.0, 0.0)
    pull.velocity = 0.02
    pull.distance_threshold = 0.001
    pull.done_when_travel = ("drawer", "drawer_housing", 0.0247, 0.027)
    commands[4].duration_s = 0.6
    commands[4].phase = "release"
    # The exit between the lip and upper guide is only about 1 mm high.
    # Lift around the round feature, clear the lip, then move beyond the guide
    # before raising the drawer farther.
    extraction = [
        ((-0.0035, 0.0, 0.009), 0.01),
        ((-0.005, 0.0, exit_height), 0.01),
        ((-0.019, 0.0, exit_height), 0.02),
        ((-0.047, 0.0, 0.030), 0.025),
    ]
    commands[4:4] = [
        pull.replace(target_offset_pos=offset, velocity=speed, done_when_travel=None)
        for offset, speed in extraction
    ]
    # Finish on drawer travel while carrying its weight; final success checks
    # that both shaft bounds are clear, including the drawer's rotation.
    commands[6].done_when_travel = ("drawer", "drawer_housing", 0.04, 0.08)
    commands[7].distance_threshold = 0.005
    for index, phase in ((0, "approach"), (1, "engage"), (2, "grip"), (9, "retreat")):
        commands[index].phase = phase
