# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES
from isaaclab_hiveboard import mdp as spot_mdp


@configclass
class ObservationsCfg:
    """Observation specifications for Spot Small Valve."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiffusionPolicyCfg(ObsGroup):
        """Concatenated observations matching the cleaned diffusion dataset."""

        ee_pose_b = ObsTerm(
            func=spot_mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "frame_name": "ee_frame"},
        )
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES, preserve_order=True
                )
            },
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES, preserve_order=True
                )
            },
        )
        object_root_pose_b = ObsTerm(
            func=spot_mdp.object_root_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("small_valve"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
