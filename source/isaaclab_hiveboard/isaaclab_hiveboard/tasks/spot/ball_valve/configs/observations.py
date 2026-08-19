from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES
from isaaclab_hiveboard import mdp as spot_mdp


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # pose = ObsTerm(func=my_mdp.rel_ee_drawer_pose)
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
