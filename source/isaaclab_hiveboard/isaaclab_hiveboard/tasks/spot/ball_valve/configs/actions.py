from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp


@configclass
class SpotIKAbsActionCfg:
    """Action specifications for the MDP."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
        ],
        body_name="arm_link_wr1",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        scale=1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.21, 0.0, -0.03),  # Tool Center Point
        ),
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.57},
        close_command_expr={"arm_f1x": -0.3},
    )


@configclass
class SpotIKRelativeActionCfg:
    """Relative TCP-pose actions for fixed-base Spot manipulation."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
        ],
        body_name="arm_link_wr1",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        # At 20 Hz, normalized actions map to 4 cm translation and 0.2 rad
        # rotation, preserving the former 40 Hz maximum task-space velocity.
        scale=(0.04, 0.04, 0.04, 0.2, 0.2, 0.2),
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.21, 0.0, -0.03),  # Tool Center Point
        ),
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.57},
        close_command_expr={"arm_f1x": -0.3},
    )
