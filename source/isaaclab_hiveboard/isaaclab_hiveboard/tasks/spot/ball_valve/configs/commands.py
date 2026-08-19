from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)


@configclass
class FramePoseCommandsCfg:
    """Command specifications for the RMP."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_wr1",
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=0.5,
        # HiveBoard limits are [-pi/2, 0]: negative rotation opens the valve.
        valve_joint_closed=0.0,
        valve_joint_open=-1.5707963267948966,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
                velocity=0.25,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                target_frame_name="lever_pivot",
                velocity=0.15,
            ),
            # GripperCommand(open_gripper=False, duration_s=0.3),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                # Fallback only; the command term uses the remaining valve error.
                angle_deg=-90,
                angular_velocity=0.3,
                gripper_open=False,
            ),
            # Hold the finished pose so the end state reads on video.
            GripperCommand(open_gripper=False, duration_s=0.075),
        ],
        body_offset=SequentialPoseCommandCfg.OffsetCfg(
            pos=(0.21, 0.0, -0.03),  # Tool Center Point
        ),
    )
