from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.tasks.spot.high_torque_valve.configs.scene import EE_TCP_OFFSET


@configclass
class FramePoseCommandsCfg:
    """Clamp the handwheel, then spin about +Z.

    Reset IK already places the TCP on ``approaching``, so the sequence starts
    at ``nut_grasp`` instead of an arrival command.
    """

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_wr1",
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        commands=[
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.02,
                target_frame_name="nut_grasp",
            ),
            RotateFrameCfg(
                frame_name="target_frame",
                target_frame_name="rotate_frame",
                angle_deg=-180,
                gripper_open=False,
            ),
            GoToFrameCfg(
                frame_name="target_frame",
                gripper_open=True,
                distance_threshold=0.03,
                target_frame_name="approaching",
            ),
            GripperCommand(open_gripper=True, duration_s=0.10),
        ],
        body_offset=SequentialPoseCommandCfg.OffsetCfg(
            pos=EE_TCP_OFFSET.pos,
            rot=EE_TCP_OFFSET.rot,
        ),
    )
