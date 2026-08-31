from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import SPOT_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    ScrewJointCouplingCfg,
    SequentialPoseCommandCfg,
)


def _turn_cycle(*, unwind_first: bool) -> list:
    """One 1.5 mm quarter-turn, optionally unwinding Spot before regrasping."""
    commands = []
    if unwind_first:
        commands.extend(
            [
                RotateFrameCfg(
                    frame_name="target_frame",
                    target_frame_name="release_frame",
                    axis=(1.0, 0.0, 0.0),
                    angle_deg=-90.0,
                    angular_velocity=1.5,
                    angle_threshold_deg=2.0,
                    gripper_open=True,
                    axis_position_override_b=(None, 0.0, -0.030),
                ),
            ]
        )
    commands.extend(
        [
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="lamp_grasp",
                gripper_open=True,
                # The reset pose is already at the bulb's accessible grasp
                # surface; avoid driving the palm through the bulb center.
                distance_threshold=0.015,
                orientation_threshold_deg=10.0,
                hold_current_orientation=True,
                position_override_b=(None, 0.0, -0.030),
            ),
            GripperCommand(open_gripper=False, duration_s=0.75),
            ScrewFrameCfg(
                frame_name="target_frame",
                target_frame_name="screw_frame",
                axis=(1.0, 0.0, 0.0),
                angle_deg=90.0,
                # Quarter turn at the USD-authored 6 mm/revolution pitch.
                axial_distance=0.0015,
                angular_velocity=1.5,
                angle_threshold_deg=2.0,
                gripper_open=False,
            ),
            # Leave the thick part of the bulb before expanding the fingers.
            GoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="release_frame",
                gripper_open=False,
                distance_threshold=0.015,
                orientation_threshold_deg=10.0,
                hold_current_orientation=True,
                position_override_b=(None, 0.0, -0.030),
            ),
            GripperCommand(open_gripper=True, duration_s=0.5),
        ]
    )
    return commands


@configclass
class FramePoseCommandsCfg:
    """Seat the lamp with sixteen 1.5 mm quarter-turns and wrist unwinds."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=SPOT_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            *_turn_cycle(unwind_first=False),
            *[
                command
                for _ in range(15)
                for command in _turn_cycle(unwind_first=True)
            ],
        ],
        body_offset=as_command_offset(SPOT_EE),
        screw_coupling=ScrewJointCouplingCfg(
            asset_name="lamp",
            pitch_m_per_revolution=0.006,
            # The lamp joint axis points opposite Spot's TCP screw axis after
            # applying the lamp's spawn rotation.
            command_joint_angle_scale=-1.0,
            lower_limit=0.0,
            upper_limit=0.024,
        ),
    )
