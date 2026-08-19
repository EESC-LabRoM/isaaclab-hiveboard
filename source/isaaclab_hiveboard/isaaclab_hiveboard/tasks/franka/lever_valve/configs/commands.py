from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Franka approach lever valve, grasp handle, rotate 90 deg, then release."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
                num_frames=30,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                target_frame_name="lever_pivot",
                num_frames=30,
            ),
            GripperCommand(open_gripper=False, num_frames=15),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                axis=(1.0, 0.0, 0.0),  # Ball-valve URDF RevoluteJoint axis
                angle_deg=-90,
                num_frames=200,
                gripper_open=False,
            ),
            GripperCommand(open_gripper=False, num_frames=15),
            GripperCommand(open_gripper=True, num_frames=15),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
                num_frames=30,
            ),
            GripperCommand(open_gripper=True, num_frames=30),
        ],
        body_offset=SequentialPoseCommandCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1034),  # Franka TCP offset
        ),
    )
