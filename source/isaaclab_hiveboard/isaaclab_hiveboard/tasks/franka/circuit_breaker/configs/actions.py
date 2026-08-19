from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp


@configclass
class FrankaIKAbsActionCfg:
    """Action specifications for Franka Emika Panda with Differential IK."""

    # ActionManager concatenates terms in declaration order. SequentialPoseCommand
    # emits [gripper, position XYZ, quaternion WXYZ], so the one-dimensional
    # gripper term must be declared before the seven-dimensional arm term.
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint.*"],
        open_command_expr={"panda_finger_joint.*": 0.04},
        close_command_expr={"panda_finger_joint.*": 0.0},
    )

    arm_action = DifferentialInverseKinematicsActionCfg(
        debug_vis=False,
        asset_name="robot",
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1034),  # Franka Tool Center Point (between finger tips)
        ),
    )
