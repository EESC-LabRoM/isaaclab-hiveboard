# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retarget a shared HiveBoard task config onto fixed-base Spot.

The counterpart of :func:`isaaclab_hiveboard.tasks.franka.common.use_franka`
for tasks authored against ANYmal. Spot and ANYmal share every scene placement
(both shoulders sit ~0.65 m up, 1 m from the object), so only the robot-specific
fields change: the articulation, the TCP profile and its UUC prim paths, the
cuRobo model, the action terms, the finger bodies and the arm joint names.
"""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_hiveboard.assets import ASSET_DIR, SPOT_EE, as_command_offset, make_ee_frame
from isaaclab_hiveboard.assets.spot.constants import ARM_JOINT_NAMES
from isaaclab_hiveboard.assets.spot.spot import (
    SPOT_ARM_NEWTON_CFG,
    SPOT_ARM_UUC_BODY_PRIM,
    SPOT_ARM_UUC_FNGR_PRIM,
    SPOT_ARM_UUC_JAW_PRIM,
    SPOT_ARM_UUC_SOURCE_PRIM,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import SpotJointPositionActionCfg

SPOT_CUROBO = {
    "robot_joint_names": list(ARM_JOINT_NAMES[:-1]),
    "robot_curobo_yaml": f"{ASSET_DIR}/spot/cumotion/spot_arm.yaml",
    "robot_urdf": f"{ASSET_DIR}/spot/spot_with_arm.urdf",
}
SPOT_FINGER_BODY_NAMES = ["arm_link_fngr", "arm_link_wr1"]


def _roll_about_tcp_x(quat_xyzw: tuple[float, float, float, float], angle: float) -> tuple[float, float, float, float]:
    """Right-multiply ``quat_xyzw`` by a rotation of ``angle`` about the TCP's own +X."""
    x1, y1, z1, w1 = quat_xyzw
    x2, y2, z2, w2 = math.sin(angle / 2.0), 0.0, 0.0, math.cos(angle / 2.0)
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def use_spot(env_cfg, pinch_frames: tuple[str, ...] = ()) -> None:
    """Swap the robot-specific parts of a shared HiveBoard task for Spot.

    Args:
        env_cfg: An ANYmal task config, after its own ``__post_init__``.
        pinch_frames: Target frames that pinch across the canonical TCP +Y.
            Spot's finger swings open along TCP +Z instead, so these frames
            are rolled -90 deg about TCP +X to put its jaws across the same
            object axis as the 2F-140 and FR3 hands.
    """
    scene = env_cfg.scene
    scene.robot = SPOT_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = make_ee_frame(SPOT_EE)
    # The UUC Spot nests its links under different prims than SPOT_EE's names.
    ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_SOURCE_PRIM
    for frame in ee_frame.target_frames:
        if frame.name == "ee_tcp":
            frame.prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_BODY_PRIM
    scene.ee_frame = ee_frame
    for sensor_name, prim in (("finger_contact", SPOT_ARM_UUC_FNGR_PRIM), ("jaw_contact", SPOT_ARM_UUC_JAW_PRIM)):
        sensor = getattr(scene, sensor_name, None)
        if sensor is not None:
            sensor.prim_path = "{ENV_REGEX_NS}/Robot/" + prim

    pose_command = env_cfg.commands.pose_command
    pose_command.body_name = SPOT_EE.body_name
    pose_command.body_offset = as_command_offset(SPOT_EE)
    for command in pose_command.commands:
        if hasattr(command, "robot_joint_names"):
            for key, value in SPOT_CUROBO.items():
                setattr(command, key, value)

    for frame in scene.target_frame.target_frames:
        if frame.name in pinch_frames:
            frame.offset = OffsetCfg(pos=frame.offset.pos, rot=_roll_about_tcp_x(frame.offset.rot, -math.pi / 2))

    env_cfg.actions = SpotJointPositionActionCfg()

    material = getattr(env_cfg.events, "robot_physics_material", None)
    if material is not None:
        material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=SPOT_FINGER_BODY_NAMES)

    for group in vars(env_cfg.observations).values():
        if not isinstance(group, ObservationGroupCfg):
            continue
        for term in vars(group).values():
            if not isinstance(term, ObservationTermCfg):
                continue
            asset_cfg = term.params.get("asset_cfg")
            if not (isinstance(asset_cfg, SceneEntityCfg) and asset_cfg.name == "robot" and asset_cfg.joint_names):
                continue
            # Spot's arm observations conventionally include the arm_f1x gripper.
            is_arm = len(asset_cfg.joint_names) > 1
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=list(ARM_JOINT_NAMES) if is_arm else ["arm_f1x"], preserve_order=True
            )
