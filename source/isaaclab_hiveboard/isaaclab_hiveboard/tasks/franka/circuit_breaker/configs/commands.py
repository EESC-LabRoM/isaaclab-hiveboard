from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
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
        body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.09,
                orientation_threshold_deg=25.0,
                canonicalize_upward=False,
                target_frame_name="approaching",
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.04,
                orientation_threshold_deg=25.0,
                canonicalize_upward=False,
                target_frame_name="lever_pivot_below",
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.04,
                orientation_threshold_deg=25.0,
                canonicalize_upward=False,
                target_frame_name="lever_pivot_above",
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=False,
                distance_threshold=0.09,
                orientation_threshold_deg=25.0,
                canonicalize_upward=False,
                target_frame_name="approaching",
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
        ],
        body_offset=as_command_offset(FRANKA_EE),
    )
