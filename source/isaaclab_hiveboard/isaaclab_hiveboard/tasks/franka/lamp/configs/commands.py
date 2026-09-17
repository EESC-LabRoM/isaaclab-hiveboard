from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    ScrewJointCouplingCfg,
    SequentialPoseCommandCfg,
)


def _unscrew_cycle(*, unwind_first: bool) -> list:
    """Pull the bulb outward by one quarter-turn, then release it safely."""
    commands = []
    if unwind_first:
        commands.append(
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="approaching",
                axis=(1.0, 0.0, 0.0),
                angle_deg=90.0,
                angular_velocity=0.75,
                angle_threshold_deg=2.0,
                gripper_open=True,
            )
        )

    commands.extend(
        [
            # Engage by translating along the lamp axis. Replanning here can
            # select another redundant-joint branch and rotate in place.
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="lamp_grasp",
                gripper_open=True,
                velocity=0.08,
                # Do not start closing while the TCP is merely near the bulb.
                # The former 10 mm tolerance advanced 6-9 mm before the
                # commanded grasp pose was reached.
                distance_threshold=0.002,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
                hold_current_orientation=True,
            ),
            GripperCommand(open_gripper=False, duration_s=0.75),
            ScrewFrameCfg(
                frame_name="target_frame",
                target_frame_name="screw_frame",
                axis=(1.0, 0.0, 0.0),
                angle_deg=-90.0,
                axial_distance=-0.0015,
                angular_velocity=0.5,
                angle_threshold_deg=2.0,
                gripper_open=False,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="approaching",
                gripper_open=False,
                velocity=0.08,
                distance_threshold=0.015,
                orientation_threshold_deg=10.0,
                canonicalize_upward=False,
                hold_current_orientation=True,
            ),
            GripperCommand(open_gripper=True, duration_s=0.5),
        ]
    )
    return commands


@configclass
class FramePoseCommandsCfg:
    """Approach, grasp, unscrew, and remove the lamp with Franka."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="approaching",
                gripper_open=True,
                distance_threshold=0.03,
                orientation_threshold_deg=15.0,
                canonicalize_upward=False,
                robot_joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            ),
            *_unscrew_cycle(unwind_first=False),
            *[
                command
                for _ in range(15)
                for command in _unscrew_cycle(unwind_first=True)
            ],
        ],
        body_offset=as_command_offset(FRANKA_EE),
        screw_coupling=ScrewJointCouplingCfg(
            asset_name="lamp",
            pitch_m_per_revolution=0.006,
            # The lamp joint coordinate has the opposite sign to Franka's
            # canonical TCP screw axis.
            command_joint_angle_scale=-1.0,
            lower_limit=0.0,
            upper_limit=0.024,
        ),
    )
