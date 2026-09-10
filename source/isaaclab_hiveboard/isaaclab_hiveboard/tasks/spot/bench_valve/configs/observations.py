# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES
from isaaclab_hiveboard import mdp as spot_mdp


@configclass
class ObservationsCfg:
    """Observations for Cartesian key following."""

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )
        finger_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w,
            params={"sensor_cfg": SceneEntityCfg("finger_contact")},
        )
        jaw_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w,
            params={"sensor_cfg": SceneEntityCfg("jaw_contact")},
        )
        finger_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("finger_contact")},
        )
        jaw_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("jaw_contact")},
        )
        wr1_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("wr1_contact")},
        )
        wr0_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("wr0_contact")},
        )
        el1_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("el1_contact")},
        )
        el0_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("el0_contact")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiffusionPolicyCfg(ObsGroup):
        """Keep the recorder HDF5 schema happy; not used for control."""

        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_JOINT_NAMES, preserve_order=True
                )
            },
        )
        object_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "ball_valve", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        valve_task_direction = ObsTerm(
            func=spot_mdp.valve_task_direction,
            params={"command_name": "pose_command"},
        )
        valve_current_angle = ObsTerm(
            func=spot_mdp.valve_current_angle,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg(
                    "ball_valve", joint_names=["RevoluteJoint"], preserve_order=True
                ),
            },
        )
        valve_goal_angle = ObsTerm(
            func=spot_mdp.valve_goal_angle,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EvaluationCfg(ObsGroup):
        tcp_pose_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
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
        valve_angle = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "ball_valve", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        valve_velocity = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "ball_valve", joint_names=["RevoluteJoint"], preserve_order=True
                )
            },
        )
        valve_task_direction = ObsTerm(
            func=spot_mdp.valve_task_direction,
            params={"command_name": "pose_command"},
        )
        valve_current_angle = ObsTerm(
            func=spot_mdp.valve_current_angle,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg(
                    "ball_valve", joint_names=["RevoluteJoint"], preserve_order=True
                ),
            },
        )
        valve_goal_angle = ObsTerm(
            func=spot_mdp.valve_goal_angle,
            params={"command_name": "pose_command"},
        )
        finger_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w,
            params={"sensor_cfg": SceneEntityCfg("finger_contact")},
        )
        jaw_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w,
            params={"sensor_cfg": SceneEntityCfg("jaw_contact")},
        )
        finger_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("finger_contact")},
        )
        jaw_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("jaw_contact")},
        )
        wr1_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("wr1_contact")},
        )
        wr0_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("wr0_contact")},
        )
        el1_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("el1_contact")},
        )
        el0_valve_force = ObsTerm(
            func=spot_mdp.contact_force_matrix_w,
            params={"sensor_cfg": SceneEntityCfg("el0_contact")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
