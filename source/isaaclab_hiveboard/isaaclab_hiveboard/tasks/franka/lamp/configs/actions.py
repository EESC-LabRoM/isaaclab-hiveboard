"""Absolute TCP pose control for Franka's lamp task."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_ik_offset
from isaaclab_hiveboard.mdp.actions.curobo_planned_ik import CuroboPlannedDifferentialInverseKinematicsActionCfg
from isaaclab_hiveboard.tasks.spot.lamp.configs.actions import SpotLampActionCfg


@configclass
class FrankaLampActionCfg(SpotLampActionCfg):
    def __post_init__(self):
        self.arm_action = CuroboPlannedDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[f"fr3_joint{i}" for i in range(1, 8)],
            body_name=FRANKA_EE.body_name,
            body_offset=as_ik_offset(FRANKA_EE),
            controller=self.arm_action.controller,
            scale=self.arm_action.scale,
        )
        self.gripper_action.joint_names = ["fr3_finger_joint.*"]
        self.gripper_action.open_command_expr = {"fr3_finger_joint.*": 0.04}
        self.gripper_action.close_command_expr = {"fr3_finger_joint.*": 0.018}
