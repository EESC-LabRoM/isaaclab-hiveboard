"""Frozen observation and action layouts for Spot ball-valve imitation.

The live ``diffusion_policy`` group is the 43-D student: proprio, object root,
handle pose in the TCP, open/close command, and last action. Train from
regenerated recordings that already contain this layout.
"""

from __future__ import annotations

PROPRIO_OBJECT_NAMES: tuple[str, ...] = (
    "ee_pos_b_x",
    "ee_pos_b_y",
    "ee_pos_b_z",
    "ee_quat_b_w",
    "ee_quat_b_x",
    "ee_quat_b_y",
    "ee_quat_b_z",
    "arm_joint_pos_0",
    "arm_joint_pos_1",
    "arm_joint_pos_2",
    "arm_joint_pos_3",
    "arm_joint_pos_4",
    "arm_joint_pos_5",
    "gripper_joint_pos",
    "arm_joint_vel_0",
    "arm_joint_vel_1",
    "arm_joint_vel_2",
    "arm_joint_vel_3",
    "arm_joint_vel_4",
    "arm_joint_vel_5",
    "gripper_joint_vel",
    "object_root_pos_b_x",
    "object_root_pos_b_y",
    "object_root_pos_b_z",
    "object_root_quat_b_w",
    "object_root_quat_b_x",
    "object_root_quat_b_y",
    "object_root_quat_b_z",
)

HANDLE_POSE_EE_NAMES: tuple[str, ...] = (
    "handle_pos_ee_x",
    "handle_pos_ee_y",
    "handle_pos_ee_z",
    "handle_quat_ee_w",
    "handle_quat_ee_x",
    "handle_quat_ee_y",
    "handle_quat_ee_z",
)

VALVE_TASK_DIRECTION_NAMES: tuple[str, ...] = ("valve_task_direction",)

LAST_ACTION_NAMES: tuple[str, ...] = (
    "last_gripper_command",
    "last_delta_pos_x_normalized",
    "last_delta_pos_y_normalized",
    "last_delta_pos_z_normalized",
    "last_delta_axis_angle_x_normalized",
    "last_delta_axis_angle_y_normalized",
    "last_delta_axis_angle_z_normalized",
)

OBSERVATION_NAMES: tuple[str, ...] = (
    PROPRIO_OBJECT_NAMES
    + HANDLE_POSE_EE_NAMES
    + VALVE_TASK_DIRECTION_NAMES
    + LAST_ACTION_NAMES
)

ACTION_NAMES: tuple[str, ...] = (
    "gripper_command",
    "delta_pos_x_normalized",
    "delta_pos_y_normalized",
    "delta_pos_z_normalized",
    "delta_axis_angle_x_normalized",
    "delta_axis_angle_y_normalized",
    "delta_axis_angle_z_normalized",
)

PROPRIO_OBJECT_DIM = len(PROPRIO_OBJECT_NAMES)
HANDLE_POSE_EE_DIM = len(HANDLE_POSE_EE_NAMES)
VALVE_TASK_DIRECTION_DIM = len(VALVE_TASK_DIRECTION_NAMES)
STUDENT_BODY_DIM = (
    PROPRIO_OBJECT_DIM + HANDLE_POSE_EE_DIM + VALVE_TASK_DIRECTION_DIM
)
OBSERVATION_DIM = len(OBSERVATION_NAMES)
ACTION_DIM = len(ACTION_NAMES)


