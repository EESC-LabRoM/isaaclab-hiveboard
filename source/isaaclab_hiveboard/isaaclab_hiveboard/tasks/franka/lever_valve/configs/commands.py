from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Franka approach lever valve, grasp handle, rotate 90 deg, then release."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                canonicalize_upward=False,
                target_frame_name="lever_pivot",
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
            GripperCommand(
                open_gripper=False,
                duration_s=0.3,
            ),
            CuroboPlannedRotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # The Franka workspace rotates the valve root 180 deg about Z,
                # so the URDF joint's local +X axis is base-frame -X.
                axis=(-1.0, 0.0, 0.0),
                angle_deg=-90,
                gripper_open=False,
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
                num_ik_seeds=16,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                canonicalize_upward=False,
                target_frame_name="approaching",
            ),
        ],
        body_offset=as_command_offset(FRANKA_EE),
    )
