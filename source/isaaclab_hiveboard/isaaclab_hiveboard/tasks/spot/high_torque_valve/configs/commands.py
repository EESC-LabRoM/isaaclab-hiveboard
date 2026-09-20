import math

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR, SPOT_EE, as_command_offset
from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)

SPOT_ARM_6 = list(ARM_JOINT_NAMES[:-1])
_CUROBO = {
    "robot_joint_names": SPOT_ARM_6,
    "robot_curobo_yaml": f"{ASSET_DIR}/spot/cumotion/spot_arm.yaml",
    "robot_urdf": f"{ASSET_DIR}/spot/spot_with_arm.urdf",
}


@configclass
class FramePoseCommandsCfg:
    """Approach the handwheel, clamp, spin, then retreat."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=SPOT_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        valve_asset_name="high_torque_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=0.0,
        valve_joint_closed=0.0,
        valve_joint_open=-math.pi,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
                velocity=0.25,
                **_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                target_frame_name="nut_grasp",
                velocity=0.15,
                **_CUROBO,
            ),
            GripperCommand(open_gripper=False, duration_s=0.3),
            CuroboPlannedRotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                angle_deg=-180,
                angular_velocity=0.3,
                angle_threshold_deg=0.25,
                gripper_open=False,
                axis=(-1.0, 0.0, 0.0),
                **_CUROBO,
            ),
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
                velocity=0.25,
                **_CUROBO,
            ),
            GripperCommand(open_gripper=True, duration_s=0.10),
        ],
        body_offset=as_command_offset(SPOT_EE),
    )
