"""Absolute TCP pose control for the Newton lamp task."""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import SPOT_EE, as_ik_offset
from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES
from isaaclab_hiveboard.mdp.pose_actions import OffsetDifferentialIKAction


@configclass
class SpotLampActionCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        class_type=OffsetDifferentialIKAction,
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES[:-1]),
        body_name=SPOT_EE.body_name,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=as_ik_offset(SPOT_EE),
        scale=1.0,
    )
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.3},
        close_command_expr={"arm_f1x": -0.6},
    )
