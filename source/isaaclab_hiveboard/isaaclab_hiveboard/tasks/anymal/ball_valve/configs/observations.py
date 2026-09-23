# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES, NEWTON_GRIPPER_JOINT_NAMES
from isaaclab_hiveboard import mdp as spot_mdp


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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
                    "robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True
                )
            },
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True
                )
            },
        )
        object_root_pose_b = ObsTerm(
            func=spot_mdp.object_root_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("ball_valve"),
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
        object_joint_vel = ObsTerm(
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EvaluationCfg(ObsGroup):
        """Named physical signals used by policy evaluation metrics."""

        ee_pose_b = ObsTerm(
            func=spot_mdp.ee_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "frame_name": "ee_frame"},
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
        object_root_pose_b = ObsTerm(
            func=spot_mdp.object_root_pose_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("ball_valve"),
            },
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

    @configclass
    class BehaviorCloningCfg(ObsGroup):
        """Per-key observations for robomimic behaviour cloning and DAgger.

        Terms stay unconcatenated: robomimic keys every modality separately in
        ``data/demo_*/obs/<key>`` and builds an encoder per key, so fusing them
        here would collapse the ``low_dim`` modality list into one blob.

        Three tiers of information, all reproducible on hardware:

        * Proprioception (``eef_*``, ``arm_joint_*``, ``gripper_pos``) comes
          from joint encoders and forward kinematics.
        * Privileged object state (``object_pos``, ``object_quat``,
          ``valve_current_angle``) is what an AprilTag on the valve supplies
          directly, so no state estimator is needed at deployment.
        * Task specification (``valve_goal_angle``,
          ``valve_task_direction``) tells the policy *which* way to turn. This
          task samples open and close episodes, so a policy without these terms
          cannot do better than guessing the direction.

        Contact forces are deliberately excluded even though the task records
        them: the 2F-140 reports no contact wrench on hardware, so a policy
        that leaned on them would not transfer.

        ``gripper_pos`` reads ``finger_joint`` alone. It is the 2F-140's motor
        joint - the rest of the parallel linkage is driven by USD constraints,
        so the remaining finger joints carry no independent information.
        """

        eef_pos = ObsTerm(
            func=spot_mdp.ee_pos_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "frame_name": "ee_frame"},
        )
        eef_quat = ObsTerm(
            func=spot_mdp.ee_quat_b,
            params={"asset_cfg": SceneEntityCfg("robot"), "frame_name": "ee_frame"},
        )
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)},
        )
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=NEWTON_GRIPPER_JOINT_NAMES, preserve_order=True)
            },
        )
        object_pos = ObsTerm(
            func=spot_mdp.object_pos_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("ball_valve"),
            },
        )
        object_quat = ObsTerm(
            func=spot_mdp.object_quat_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("ball_valve"),
            },
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
        valve_task_direction = ObsTerm(
            func=spot_mdp.valve_task_direction,
            params={"command_name": "pose_command"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    bc: BehaviorCloningCfg = BehaviorCloningCfg()
