# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, as_command_offset
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import SequentialPoseCommandCfg
from isaaclab_hiveboard.mdp.commands.skills import SkillSet
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO

ANYMAL = SkillSet(**ANYMAL_CUROBO)


@configclass
class FramePoseCommandsCfg:
    """cuRobo joint plans for the ball-valve sequence."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name=ANYMAL_EE.body_name,
        resampling_time_range=(1e6, 1e6),
        debug_vis=False,
        output_joint_positions=True,
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=0.0,
        # HiveBoard limits are [-pi/2, 0]: negative rotation opens the valve.
        valve_joint_closed=0.0,
        valve_joint_open=-1.5707963267948966,
        valve_min_delta_rad=0.35,
        valve_ee_joint_angle_scale=1.0,
        commands=[
            ANYMAL.approach("approaching"),
            ANYMAL.engage("lever_pivot"),
            ANYMAL.grip(duration_s=0.3),
            ANYMAL.turn(angle_deg=-90, angular_velocity=0.3, angle_threshold_deg=0.25),
            ANYMAL.hold("actuate", duration_s=0.5),
        ],
        body_offset=as_command_offset(ANYMAL_EE),
    )
