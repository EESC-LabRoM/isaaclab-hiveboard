# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES
from isaaclab_hiveboard import mdp as spot_mdp


@configclass
class ObservationsCfg:
    """Observations for Cartesian key following.

    Unlike the Spot bench env, there are no verified per-link contact-sensor
    prim paths along the DynaArm (Spot's ``wr0``/``wr1``/``el0``/``el1``
    sensors have no established ANYmal analog); only the 2F-140 pad contacts
    are wired up here.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiffusionPolicyCfg(ObsGroup):
        """Keep the recorder HDF5 schema happy; not used for control."""

        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        object_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"], preserve_order=True)},
        )
        valve_task_direction = ObsTerm(
            func=spot_mdp.valve_task_direction,
            params={"command_name": "pose_command"},
        )
        valve_current_angle = ObsTerm(
            func=spot_mdp.valve_current_angle,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"], preserve_order=True),
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
        tcp_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        valve_angle = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"], preserve_order=True)},
        )
        valve_velocity = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"], preserve_order=True)},
        )
        valve_task_direction = ObsTerm(
            func=spot_mdp.valve_task_direction,
            params={"command_name": "pose_command"},
        )
        valve_current_angle = ObsTerm(
            func=spot_mdp.valve_current_angle,
            params={
                "command_name": "pose_command",
                "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"], preserve_order=True),
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    evaluation: EvaluationCfg = EvaluationCfg()

    @configclass
    class CameraObsCfg(ObsGroup):
        """Fixed scene-camera RGB frames (unnormalized uint8) for video recording."""

        scene_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("scene_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    camera: CameraObsCfg = CameraObsCfg()
