from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)

FRANKA_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


@configclass
class FramePoseCommandsCfg:
    """Approach/open the cover, move above the button, and press it fully."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="cover_clearance",
                gripper_open=True,
                distance_threshold=0.01,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
                robot_joint_names=FRANKA_JOINTS,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="cover_grasp",
                gripper_open=True,
                velocity=0.08,
                distance_threshold=0.006,
                orientation_threshold_deg=5.0,
                canonicalize_upward=False,
                hold_current_orientation=True,
            ),
            GripperCommand(open_gripper=False, duration_s=0.25),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="cover_hinge",
                axis=(0.0, 0.0, 1.0),
                angle_deg=130.0,
                angular_velocity=0.20,
                gripper_open=False,
                angle_threshold_deg=1.0,
            ),
            # Keep the final Cartesian target while the impedance controller
            # catches up, so the physically grasped lid reaches the full arc.
            GripperCommand(open_gripper=False, duration_s=0.75),
            GripperCommand(open_gripper=True, duration_s=0.25),
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="post_cover_clearance",
                gripper_open=True,
                velocity=0.15,
                distance_threshold=0.015,
                orientation_threshold_deg=15.0,
                canonicalize_upward=False,
                hold_current_orientation=True,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="button_clearance",
                gripper_open=True,
                velocity=0.15,
                distance_threshold=0.012,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
            ),
            GripperCommand(open_gripper=False, duration_s=0.2),
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="button_pressed",
                gripper_open=False,
                velocity=0.07,
                # Once the plunger bottoms out at -10 mm, contact holds the TCP
                # about 14 mm above the authored through-surface target.
                distance_threshold=0.016,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
                hold_current_orientation=True,
            ),
        ],
        body_offset=as_command_offset(FRANKA_EE),
    )
