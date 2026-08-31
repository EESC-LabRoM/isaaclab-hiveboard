# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import (
    DYNAARM_EE_LINK,
    DYNAARM_JOINT_NAMES,
    DYNAARM_MOUNT_POS,
    DYNAARM_MOUNT_ROT,
    DYNAARM_URDF,
)
from isaaclab_hiveboard.mdp.events import ResetDynaarmToFrameEvent


@configclass
class AnymalBallValveEventCfg:
    """Default scene reset, then IK the DynaArm TCP onto the approach frame."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_eef = EventTerm(
        func=ResetDynaarmToFrameEvent,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(DYNAARM_JOINT_NAMES), preserve_order=True
            ),
            "frame_name": "target_frame",
            "target_frame_name": "approaching",
            "command_name": "pose_command",
            "robot_urdf": DYNAARM_URDF,
            "robot_root_link": "arm_mount",
            "robot_ee_link": DYNAARM_EE_LINK,
            "mount_pos": DYNAARM_MOUNT_POS,
            "mount_rot": DYNAARM_MOUNT_ROT,
        },
    )
