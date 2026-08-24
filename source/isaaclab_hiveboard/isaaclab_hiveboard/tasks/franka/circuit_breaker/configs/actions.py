from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_hiveboard.assets import FRANKA_EE, as_ik_offset
from isaaclab_hiveboard.mdp.actions.curobo_planned_ik import (
    CuroboPlannedDifferentialInverseKinematicsActionCfg,
)


@configclass
class FrankaIKAbsActionCfg:
    """Action specifications for Franka Research 3 with Differential IK."""

    # ActionManager concatenates terms in declaration order. SequentialPoseCommand
    # emits [gripper, position XYZ, quaternion WXYZ], so the one-dimensional
    # gripper term must be declared before the seven-dimensional arm term.
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04},
        close_command_expr={"fr3_finger_joint.*": 0.0},
    )

    arm_action = CuroboPlannedDifferentialInverseKinematicsActionCfg(
        debug_vis=False,
        asset_name="robot",
        joint_names=[
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ],
        body_name=FRANKA_EE.body_name,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.1},
        ),
        scale=1.0,
        body_offset=as_ik_offset(FRANKA_EE),
    )
