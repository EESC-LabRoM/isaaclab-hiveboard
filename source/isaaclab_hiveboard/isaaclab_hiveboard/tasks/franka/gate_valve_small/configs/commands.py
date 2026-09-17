from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.progress import GateValvePoseCommand

# Motion tuning. Linear values are metres/second; angular values are
# radians/second. Franka's generic GoToFrame default is 0.20 m/s.
LINEAR_VELOCITY = 0.15
ANGULAR_VELOCITY = 0.60
# The physical wheel trails the TCP by about 2 degrees at this speed because
# of contact compliance. This calibrated command produces a measured 90-degree
# wheel turn; completion still uses actual stem travel rather than this target.
TURN_COMMAND_DEG = 92.0


def _quarter_turn() -> list:
    """Engage, turn, release, lift, then unwind safely above the stem."""
    return [
        GoToFrameCfg(
            frame_name="target_frame", target_frame_name="nut_grasp",
            gripper_open=True, velocity=LINEAR_VELOCITY, distance_threshold=0.002,
            canonicalize_upward=False,
        ),
        GripperCommand(open_gripper=False, duration_s=0.75),
        RotateFrameCfg(
            frame_name="target_frame", target_frame_name="rotate_frame",
            axis=(0.0, -1.0, 0.0), angle_deg=TURN_COMMAND_DEG,
            angular_velocity=ANGULAR_VELOCITY, angle_threshold_deg=0.0, gripper_open=False,
        ),
        # Let the physical wrist catch the final target before releasing.
        GripperCommand(open_gripper=False, duration_s=0.5),
        GripperCommand(open_gripper=True, duration_s=0.5),
        GoToFrameCfg(
            frame_name="target_frame", target_frame_name="approaching",
            gripper_open=True, velocity=LINEAR_VELOCITY, distance_threshold=0.005,
            canonicalize_upward=False, hold_current_orientation=True,
        ),
        # Reset orientation only after lifting clear of the handwheel.
        GoToFrameCfg(
            frame_name="target_frame", target_frame_name="approaching",
            gripper_open=True, distance_threshold=0.005,
            orientation_threshold_deg=2.0, canonicalize_upward=False,
        ),
    ]


@configclass
class FramePoseCommandsCfg:
    """Open from the closed zero position with four positive quarter-turns."""

    pose_command = SequentialPoseCommandCfg(
        class_type=GateValvePoseCommand,
        asset_name="robot", body_name=FRANKA_EE.body_name,
        resampling_time_range=(1e6, 1e6), debug_vis=False,
        body_offset=as_command_offset(FRANKA_EE),
        commands=[
            GoToFrameCfg(
                frame_name="target_frame", target_frame_name="approaching",
                gripper_open=True, velocity=LINEAR_VELOCITY, distance_threshold=0.005,
                orientation_threshold_deg=2.0, canonicalize_upward=False,
            ),
            *[command for _ in range(4) for command in _quarter_turn()],
        ],
    )
