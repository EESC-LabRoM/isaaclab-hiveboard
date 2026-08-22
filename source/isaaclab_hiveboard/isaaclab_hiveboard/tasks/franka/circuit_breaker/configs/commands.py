from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Franka approach circuit breaker lever, flip switch down, then hold/retreat."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="fr3_hand",
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
                target_frame_name="lever_pivot_below",
            ),
            GripperCommand(open_gripper=False, duration_s=0.3),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.02,
                target_frame_name="lever_pivot_above",
            ),
            GripperCommand(open_gripper=False, duration_s=0.375),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.03,
                target_frame_name="approaching",
            ),
            GripperCommand(open_gripper=True, duration_s=0.75),
        ],
        body_offset=SequentialPoseCommandCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1034),  # Franka TCP offset
            # rot=(0.7071068, 0.0, -0.7071068, 0.0),
        ),
    )
