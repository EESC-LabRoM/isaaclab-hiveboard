# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Retarget a shared HiveBoard task config onto the Franka Research 3.

The Spot and ANYmal tasks share one scene per HiveBoard object and differ only
in robot-specific fields: the articulation, the TCP profile, the cuRobo model,
the action terms, the finger bodies and the arm joint names. :func:`use_franka`
rewrites exactly those fields, so each Franka task is a thin subclass of the
matching ANYmal task instead of another copy of it.
"""

from __future__ import annotations

import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ASSET_DIR, FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, as_command_offset, make_ee_frame

FRANKA_ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FRANKA_FINGER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]
FRANKA_FINGER_BODY_NAMES = ["fr3_leftfinger", "fr3_rightfinger"]

FRANKA_CUROBO = {
    "robot_joint_names": FRANKA_ARM_JOINT_NAMES,
    "robot_curobo_yaml": f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
    "robot_urdf": f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
}

# Integrate the stiff drives in MJWarp's implicit solver. Explicit PD at the
# 5 ms task timestep saturates the arm torques and makes the fingers oscillate
# even under a constant open command.
ARM_STIFFNESS = 2000.0
ARM_DAMPING = 200.0
FRANKA_NEWTON_ACTUATORS = {
    "fr3_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[1-4]"],
        effort_limit_sim=87.0,
        stiffness=ARM_STIFFNESS,
        damping=ARM_DAMPING,
        armature=0.1,
    ),
    "fr3_forearm": ImplicitActuatorCfg(
        joint_names_expr=["fr3_joint[5-7]"],
        effort_limit_sim=12.0,
        stiffness=ARM_STIFFNESS,
        damping=ARM_DAMPING,
        armature=0.05,
    ),
    # Stiff enough to squeeze a handle: closing on a 1 cm part leaves a 5 mm
    # position error per finger, which must still reach the FR3 hand's
    # 70 N continuous grasp force. The damping caps the closing speed near
    # the real hand's ~0.07 m/s so the jaws do not slam light handwheels.
    # Both fingers stay driven: fr3.usd's finger mimic becomes a soft MuJoCo
    # equality in Newton, and a passive follower on it swings past its limits.
    "fr3_hand": ImplicitActuatorCfg(
        joint_names_expr=["fr3_finger_joint.*"],
        effort_limit_sim=70.0,
        stiffness=2e4,
        damping=1e3,
        armature=0.01,
    ),
}

FRANKA_NEWTON_CFG = FRANKA_FR3_HIGH_PD_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=FRANKA_FR3_HIGH_PD_CFG.spawn.replace(activate_contact_sensors=True),
    actuators=FRANKA_NEWTON_ACTUATORS,
)
"""Table-mounted FR3 at the env origin, facing +X."""

# The shared scenes place objects for Spot/ANYmal, whose arm shoulders sit
# about 0.65 m up and 1 m back. Shifting the object by this amount puts the
# horizontal-approach targets ~0.6 m forward and ~0.4 m up. Closer than that,
# the FR3 folds against its joint4/joint6 limits to keep the TCP level and
# cuRobo cannot plan the short grasp legs.
FRANKA_SCENE_SHIFT = (-0.30, 0.0, -0.25)

# Object-root randomization the FR3 can still reach. The legged robots use
# +-0.2/0.3 m and +-30 deg, which puts targets outside the FR3's 0.855 m reach.
FRANKA_OBJECT_POSE_RANGE = {
    "x": (-0.05, 0.05),
    "y": (-0.10, 0.10),
    "z": (-0.08, 0.08),
    "roll": (-math.pi / 18, math.pi / 18),
    "pitch": (-math.pi / 18, math.pi / 18),
    "yaw": (-math.pi / 12, math.pi / 12),
}


@configclass
class FrankaJointPositionActionCfg:
    """Absolute FR3 joint positions from cuRobo waypoints, plus a binary hand."""

    arm_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=FRANKA_ARM_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger_joint.*"],
        open_command_expr={"fr3_finger_joint.*": 0.04},
        close_command_expr={"fr3_finger_joint.*": 0.0},
    )


def use_franka(
    env_cfg,
    object_name: str,
    root_reset_event: str | None = None,
    scene_shift: tuple[float, float, float] = FRANKA_SCENE_SHIFT,
) -> None:
    """Swap the robot-specific parts of a shared HiveBoard task for the FR3.

    Args:
        env_cfg: A Spot or ANYmal task config, after its own ``__post_init__``.
        object_name: Scene entity of the HiveBoard object, shifted into reach.
        root_reset_event: Event term that randomizes the object root. Its pose
            range is narrowed to :data:`FRANKA_OBJECT_POSE_RANGE` unless it is
            already all zeros (the ``_PLAY`` variants).
        scene_shift: Object offset from the shared scene's spawn. Objects that
            protrude further toward the robot need a smaller -X shift.
    """
    scene = env_cfg.scene
    scene.robot = FRANKA_NEWTON_CFG
    scene.ee_frame = make_ee_frame(FRANKA_EE)
    for sensor_name, body in zip(("finger_contact", "jaw_contact"), FRANKA_FINGER_BODY_NAMES):
        sensor = getattr(scene, sensor_name, None)
        if sensor is not None:
            sensor.prim_path = "{ENV_REGEX_NS}/Robot/" + body

    obj = getattr(scene, object_name)
    obj.init_state.pos = tuple(p + d for p, d in zip(obj.init_state.pos, scene_shift))

    pose_command = env_cfg.commands.pose_command
    pose_command.body_name = FRANKA_EE.body_name
    pose_command.body_offset = as_command_offset(FRANKA_EE)
    for command in pose_command.commands:
        if hasattr(command, "robot_joint_names"):
            for key, value in FRANKA_CUROBO.items():
                setattr(command, key, value)

    env_cfg.actions = FrankaJointPositionActionCfg()

    material = getattr(env_cfg.events, "robot_physics_material", None)
    if material is not None:
        material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=FRANKA_FINGER_BODY_NAMES)
    if root_reset_event is not None:
        reset = getattr(env_cfg.events, root_reset_event)
        if any(bound != 0.0 for bounds in reset.params["pose_range"].values() for bound in bounds):
            reset.params["pose_range"] = dict(FRANKA_OBJECT_POSE_RANGE)

    for group in vars(env_cfg.observations).values():
        if not isinstance(group, ObservationGroupCfg):
            continue
        for term in vars(group).values():
            if not isinstance(term, ObservationTermCfg):
                continue
            asset_cfg = term.params.get("asset_cfg")
            if not (isinstance(asset_cfg, SceneEntityCfg) and asset_cfg.name == "robot" and asset_cfg.joint_names):
                continue
            # Anything that is not the arm is the gripper (e.g. ANYmal's
            # ``finger_joint``); the FR3 hand has two mirrored finger joints.
            is_arm = len(asset_cfg.joint_names) > 1
            term.params["asset_cfg"] = SceneEntityCfg(
                "robot",
                joint_names=FRANKA_ARM_JOINT_NAMES if is_arm else FRANKA_FINGER_JOINT_NAMES,
                preserve_order=True,
            )


def shift_target_frames(env_cfg, names: tuple[str, ...], dx: float) -> None:
    """Move the named target frames by ``dx`` along the object's outward +X.

    The FR3 TCP is at the fingertips rather than mid-pad, so its grasp frames
    sit deeper (negative ``dx``) than the shared ANYmal/Spot ones.
    """
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in names:
            x, y, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x + dx, y, z), rot=frame.offset.rot)
