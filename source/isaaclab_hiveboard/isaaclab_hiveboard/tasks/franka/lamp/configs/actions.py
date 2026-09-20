"""Absolute TCP pose control for Franka's lamp task."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, as_ik_offset
from isaaclab_hiveboard.tasks.spot.lamp.configs.actions import SpotLampActionCfg


@configclass
class FrankaLampActionCfg(SpotLampActionCfg):
    def __post_init__(self):
        self.arm_action.joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.arm_action.body_name = FRANKA_EE.body_name
        self.arm_action.body_offset = as_ik_offset(FRANKA_EE)
        self.gripper_action.joint_names = ["fr3_finger_joint.*"]
        self.gripper_action.open_command_expr = {"fr3_finger_joint.*": 0.04}
        self.gripper_action.close_command_expr = {"fr3_finger_joint.*": 0.018}
