from isaaclab.utils import configclass
from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg, GripperCommand, RotateFrameCfg, ScrewFrameCfg,
    ScrewJointCouplingCfg, SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.spot.lamp.configs.commands import _turn_cycle


@configclass
class FramePoseCommandsCfg:
    """Approach, grasp, and seat the lamp with Franka."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot", body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6), debug_vis=False,
        commands=[
            CuroboPlannedGoToFrameCfg(frame_name="target_frame", target_frame_name="approaching", gripper_open=True, distance_threshold=0.09, orientation_threshold_deg=25.0, canonicalize_upward=False, robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)]),
            CuroboPlannedGoToFrameCfg(frame_name="target_frame", target_frame_name="lamp_grasp", gripper_open=True, distance_threshold=0.04, orientation_threshold_deg=25.0, canonicalize_upward=False, robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)]),
            GripperCommand(open_gripper=False, duration_s=0.75),
            ScrewFrameCfg(frame_name="target_frame", target_frame_name="screw_frame", axis=(1.0, 0.0, 0.0), angle_deg=90.0, axial_distance=0.0015, angular_velocity=1.0, angle_threshold_deg=2.0, gripper_open=False),
            CuroboPlannedGoToFrameCfg(frame_name="target_frame", target_frame_name="release_frame", gripper_open=False, distance_threshold=0.04, orientation_threshold_deg=25.0, canonicalize_upward=False, robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)]),
            GripperCommand(open_gripper=True, duration_s=0.5),
        ], body_offset=as_command_offset(FRANKA_EE),
        screw_coupling=ScrewJointCouplingCfg(asset_name="lamp", pitch_m_per_revolution=0.006, command_joint_angle_scale=-1.0, lower_limit=0.0, upper_limit=0.024),
    )
