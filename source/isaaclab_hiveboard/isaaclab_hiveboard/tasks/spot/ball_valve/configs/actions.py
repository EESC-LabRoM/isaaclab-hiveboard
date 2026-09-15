from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES


@configclass
class SpotJointPositionActionCfg:
    """Absolute arm joint positions from the sequential command, plus a binary gripper."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES[:-1]),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.57},
        close_command_expr={"arm_f1x": -0.3},
    )


# @configclass
# class SpotIKRelativeActionCfg:
#     """Relative TCP-pose actions for fixed-base Spot manipulation."""
#
#     arm_action = DifferentialInverseKinematicsActionCfg(
#         class_type=OffsetDifferentialIKAction,
#         asset_name="robot",
#         joint_names=[
#             "arm_sh0",
#             "arm_sh1",
#             "arm_el0",
#             "arm_el1",
#             "arm_wr0",
#             "arm_wr1",
#         ],
#         body_name=SPOT_EE.body_name,
#         controller=DifferentialIKControllerCfg(
#             command_type="pose", use_relative_mode=True, ik_method="dls"
#         ),
#         # At 20 Hz, normalized actions map to 4 cm translation and 0.2 rad
#         # rotation, preserving the former 40 Hz maximum task-space velocity.
#         scale=(0.04, 0.04, 0.04, 0.2, 0.2, 0.2),
#         body_offset=as_ik_offset(SPOT_EE),
#     )
#
#     gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
#         asset_name="robot",
#         joint_names=["arm_f1x"],
#         open_command_expr={"arm_f1x": -1.57},
#         close_command_expr={"arm_f1x": -0.3},
#     )
