from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard import mdp as spot_mdp
from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES


@configclass
class ObservationsCfg:
    """Observation specifications for the Spot lamp task."""

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "pose_command"}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiffusionPolicyCfg(ObsGroup):
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
                "object_cfg": SceneEntityCfg("lamp"),
            },
        )
        lamp_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "lamp",
                    joint_names=["RevoluteJoint", "PrismaticJoint"],
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
