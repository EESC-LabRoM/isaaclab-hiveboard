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
    """Observation specifications for Spot hidden-button."""

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
        """Concatenated observations for a future diffusion dataset."""

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
                "object_cfg": SceneEntityCfg("button"),
            },
        )
        lid_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        lid_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        button_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["PrismaticJoint"], preserve_order=True
                )
            },
        )
        button_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["PrismaticJoint"], preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DiagnosticsCfg(ObsGroup):
        """Unconcatenated signals for waypoint tuning."""

        command_index = ObsTerm(
            func=spot_mdp.sequential_command_index,
            params={"command_name": "pose_command"},
        )
        lid_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        button_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button", joint_names=["PrismaticJoint"], preserve_order=True
                )
            },
        )
        ee_pos = ObsTerm(func=spot_mdp.ee_pos)
        ee_to_frames = ObsTerm(
            func=spot_mdp.ee_to_target_frame_distances,
            params={"ee_frame": "ee_frame", "frame_name": "target_frame"},
        )
        frame_pos_w = ObsTerm(
            func=spot_mdp.target_frame_positions_w,
            params={"frame_name": "target_frame"},
        )
        body_pos_w = ObsTerm(
            func=spot_mdp.asset_body_positions_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "button",
                    body_names=["World", "lid_pivot", "button_pivot"],
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    diagnostics: DiagnosticsCfg = DiagnosticsCfg()
