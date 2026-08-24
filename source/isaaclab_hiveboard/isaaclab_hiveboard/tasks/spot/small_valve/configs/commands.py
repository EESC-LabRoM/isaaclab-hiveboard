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
    """Approach the small handwheel, clamp, spin, then retreat."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=SPOT_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                target_frame_name="nut_grasp",
            ),
            GripperCommand(open_gripper=True, duration_s=0.3),
            GripperCommand(open_gripper=False, duration_s=0.3),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                angle_deg=-90,
                gripper_open=False,
            ),
            GripperCommand(open_gripper=False, duration_s=0.375),
            GripperCommand(open_gripper=True, duration_s=0.375),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
            ),
            GripperCommand(open_gripper=True, duration_s=0.75),
        ],
        body_offset=as_command_offset(SPOT_EE),
    )
