# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-D (Nucleus USD) with a Duatic DynaArm and Isaac Lab's Robotiq 2F-140.

The quadruped comes from Isaac Lab's ANYmal-D USD. The DynaArm is the committed
arm-only URDF (meshes in ``dependencies/duatic_dynaarm``), spawned under the
robot prim and welded to ``base``. The gripper is the same Nucleus 2F-140
payload that Isaac Lab selects on the UR10e with
``variants = {"Gripper": "Robotiq_2f_140"}``.
"""

from __future__ import annotations

import os
import re
from contextlib import nullcontext
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone, create_prim, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from isaaclab_hiveboard.assets.hiveboard import ASSET_DIR

ARM_PRIM = "dynaarm"
GRIPPER_PRIM = "robotiq_2f_140"
DYNAARM_URDF = os.path.join(ASSET_DIR, "anymal", "urdf", "dynaarm.urdf")
DYNAARM_USD_DIR = os.path.join(ASSET_DIR, "anymal", "usd")
# Same asset Isaac Lab mounts via the UR10e ``Robotiq_2f_140`` gripper variant.
ISAAC_ROBOTIQ_2F140_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Robotiq/2F-140/Robotiq_2F_140_physics_edit.usd"

# Top of the ANYmal-D trunk in the ``base`` frame. 180 deg yaw points the
# DynaArm's zero-config reach along robot +X (forward), not -X (aft).
DYNAARM_MOUNT_POS = (0.0, 0.0, 0.12)
DYNAARM_MOUNT_ROT = (0.0, 0.0, 0.0, 1.0)  # (w, x, y, z)

DYNAARM_JOINT_NAMES = [
    "dynaarm_shoulder_rotation",
    "dynaarm_shoulder_flexion",
    "dynaarm_elbow_flexion",
    "dynaarm_forearm_rotation",
    "dynaarm_wrist_flexion",
    "dynaarm_wrist_rotation",
]

# Nucleus 2F-140 palm. Fall back to older USD names if the payload is renamed.
ROBOTIQ_GRIPPER_ROOT = "robotiq_base_link"
# Arm-only URDF flange; the Nucleus palm is welded here with identity.
DYNAARM_EE_LINK = "dynaarm_flange"
ROBOTIQ_GRIPPER_ROOT_ALIASES = (
    "robotiq_base_link",
    "robotiq_arg2f_base_link",
    "robotiq_arg2f_140_model",
)
ROBOTIQ_DRIVE_JOINT = "finger_joint"
ROBOTIQ_JOINT_GEAR = {
    ROBOTIQ_DRIVE_JOINT: 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_finger_joint": -1.0,
}
ROBOTIQ_OPEN_Q = 0.0
# UR10e 2F-140 finger_joint upper limit is 0.7 rad (~40 deg).
ROBOTIQ_CLOSE_Q = 0.7


def robotiq_joint_targets(q: float) -> dict[str, float]:
    """Map a 2F-140 opening ``q`` (0 open, 0.7 closed) onto driven joints.

    Inner fingers track ``-q`` so the pads translate in parallel (Isaac Lab
    UR10e 2F-140 grasp setter).
    """
    return {name: gear * q for name, gear in ROBOTIQ_JOINT_GEAR.items()}


# ``finger_joint`` plus inner fingers at ``-q`` (Isaac Lab UR10e 2F-140 grasp).
# ``right_outer_knuckle_joint`` already mimics ``finger_joint`` on the USD.
ROBOTIQ_DRIVE_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[f"^{ROBOTIQ_DRIVE_JOINT}$"],
    effort_limit_sim=40.0,
    velocity_limit_sim=1.5,
    stiffness=80.0,
    damping=4.0,
    friction=0.0,
    armature=0.0,
)
ROBOTIQ_FINGER_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=20.0,
    velocity_limit_sim=1.5,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
    armature=0.0,
)
ROBOTIQ_PASSIVE_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)

ROBOTIQ_INIT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_finger_pad_joint": 0.0,
    ".*_outer_finger_joint": 0.0,
}


def _find_named_prim(root, name: str):
    from pxr import Usd

    if not root.IsValid():
        return Usd.Prim()
    if root.GetName() == name:
        return root
    for child in root.GetChildren():
        found = _find_named_prim(child, name)
        if found.IsValid():
            return found
    return Usd.Prim()


def _iter_prims(root):
    """Walk ``root`` including instance proxies (Nucleus payloads)."""
    from pxr import Usd

    if not root.IsValid():
        return
    predicate = Usd.PrimAllPrimsPredicate
    traverse = getattr(Usd, "TraverseInstanceProxies", None)
    if callable(traverse):
        predicate = traverse(Usd.PrimAllPrimsPredicate)
    yield from Usd.PrimRange(root, predicate)


def _strip_articulation_roots(prim) -> None:
    """Drop nested ArticulationRoot APIs so the payload joins ANYmal-D's tree."""
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = prim.GetStage()
    session = stage.GetSessionLayer() if stage is not None else None
    ctx = Usd.EditContext(stage, session) if session is not None else nullcontext()
    with ctx:
        for child in list(_iter_prims(prim)):
            if not child.IsValid():
                continue
            if child.HasAPI(UsdPhysics.ArticulationRootAPI):
                child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if child.HasAPI(PhysxSchema.PhysxArticulationAPI):
                child.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def _strip_physics_scenes(root) -> None:
    """Nucleus robot assets sometimes author a PhysicsScene; drop it when nesting."""
    stage = root.GetStage()
    for prim in list(_iter_prims(root)):
        if prim.IsValid() and prim.GetTypeName() == "PhysicsScene":
            stage.RemovePrim(prim.GetPath())


def _gripper_base_prim(root):
    """Locate the 2F-140 palm rigid body under ``root``."""
    from pxr import Usd

    for name in ROBOTIQ_GRIPPER_ROOT_ALIASES:
        found = _find_named_prim(root, name)
        if found.IsValid():
            return found
    return Usd.Prim()


def _usd_stale(usd_dir: str, usd_name: str, urdf_path: str, *, mimic_to_normal: bool) -> bool:
    usd_path = os.path.join(usd_dir, usd_name)
    if not os.path.exists(usd_path) or os.path.getmtime(urdf_path) > os.path.getmtime(usd_path):
        return True
    cfg_path = os.path.join(usd_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        return True
    with open(cfg_path, encoding="utf-8") as handle:
        text = handle.read()
    flag = f"convert_mimic_joints_to_normal_joints: {str(mimic_to_normal).lower()}"
    if flag not in text:
        return True
    return f"usd_file_name: {usd_name}" not in text


def _duatic_description_dir() -> Path:
    """Locate ``duatic_dynaarm_description`` by walking up from this file."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "dependencies" / "duatic_dynaarm" / "duatic_dynaarm_description"
        if candidate.is_dir():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "Could not find dependencies/duatic_dynaarm/duatic_dynaarm_description. "
        "Init the duatic_dynaarm git submodule."
    )


def _resolved_arm_urdf_path() -> str:
    """Write a converter copy of the DynaArm URDF with mesh paths on this machine.

    The committed URDF uses ``package://duatic_dynaarm_description/...``. Isaac Lab
    2.3 does not resolve that scheme, so the USD converter gets absolute paths
    under the submodule. pytorch-kinematics only needs the joint tree.
    """
    os.makedirs(DYNAARM_USD_DIR, exist_ok=True)
    mesh_dir = _duatic_description_dir() / "meshes" / "corydoras12"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"DynaArm meshes not found at '{mesh_dir}'.")
    src = Path(DYNAARM_URDF).read_text(encoding="utf-8")

    def _rewrite(match: re.Match[str]) -> str:
        name = Path(match.group(1)).name
        return f'filename="{mesh_dir / name}"'

    dst = re.sub(r'filename="([^"]+)"', _rewrite, src)
    out_path = os.path.join(DYNAARM_USD_DIR, "dynaarm.urdf")
    if not os.path.exists(out_path) or Path(out_path).read_text(encoding="utf-8") != dst:
        Path(out_path).write_text(dst, encoding="utf-8")
    return out_path


def _arm_usd_path() -> str:
    os.makedirs(DYNAARM_USD_DIR, exist_ok=True)
    urdf_path = _resolved_arm_urdf_path()
    force = _usd_stale(DYNAARM_USD_DIR, "dynaarm.usd", DYNAARM_URDF, mimic_to_normal=False)
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=DYNAARM_USD_DIR,
        usd_file_name="dynaarm.usd",
        force_usd_conversion=force,
        make_instanceable=False,
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
    )
    return UrdfConverter(converter_cfg).usd_path


def _world_matrix(prim):
    from pxr import UsdGeom

    return UsdGeom.XformCache().GetLocalToWorldTransform(prim)


def _set_world_matrix(prim, world_mat) -> None:
    """Author translate+orient so ``prim`` matches ``world_mat``."""
    from pxr import Gf, UsdGeom

    parent = prim.GetParent()
    parent_world = _world_matrix(parent) if parent.IsValid() else Gf.Matrix4d(1.0)
    local = world_mat * parent_world.GetInverse()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(local.ExtractTranslation())
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(local.ExtractRotationQuat())


def _weld_fixed(stage, parent_body, child_body, joint_path: str) -> None:
    """Identity fixed joint: ``child_body`` sits on ``parent_body``."""
    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([parent_body.GetPath()])
    joint.CreateBody1Rel().SetTargets([child_body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _attach_isaac_robotiq(robot_prim_path: str, arm_root) -> None:
    """Reference Isaac Lab's 2F-140 USD and weld its palm onto ``dynaarm_flange``.

    The Nucleus gripper spawns at the robot origin. Welding it there with a
    world-space fixed joint leaves a long lever from the flange, so the 2F-140
    floats next to the arm. Snap the palm onto the flange, then lock identity.
    """
    stage = get_current_stage()
    gripper_prim_path = f"{robot_prim_path}/{GRIPPER_PRIM}"
    # Keep the Nucleus URI so relative payloads (parts/, configuration/) resolve.
    create_prim(gripper_prim_path, usd_path=ISAAC_ROBOTIQ_2F140_USD, stage=stage)
    gripper_root = stage.GetPrimAtPath(gripper_prim_path)
    if not gripper_root.IsValid():
        raise RuntimeError(f"Failed to spawn Isaac Lab 2F-140 at '{gripper_prim_path}'.")
    gripper_root.SetInstanceable(False)
    _strip_articulation_roots(gripper_root)
    _strip_physics_scenes(gripper_root)

    flange = _find_named_prim(arm_root, "dynaarm_flange")
    palm = _gripper_base_prim(gripper_root)
    if not flange.IsValid():
        raise RuntimeError(f"DynaArm 'dynaarm_flange' not found under '{arm_root.GetPath()}'.")
    if not palm.IsValid():
        raise RuntimeError(
            f"Isaac Lab 2F-140 palm not found. Looked for {ROBOTIQ_GRIPPER_ROOT_ALIASES} under '{gripper_prim_path}'."
        )
    rel = str(palm.GetPath())[len(robot_prim_path) + 1 :]
    expected = f"{GRIPPER_PRIM}/{palm.GetName()}"
    if rel != expected:
        raise RuntimeError(
            f"Isaac Lab 2F-140 palm is at Robot/{rel}, expected Robot/{expected}. "
            "Update ANYMAL_EE.body_name / body_prim to match."
        )

    # USD row-vector: M_palm_world = M_palm_from_root * M_root_world.
    # Keep M_palm_from_root and put the palm on the flange.
    M_palm_world = _world_matrix(palm)
    M_root_world = _world_matrix(gripper_root)
    M_flange_world = _world_matrix(flange)
    M_palm_from_root = M_palm_world * M_root_world.GetInverse()
    _set_world_matrix(gripper_root, M_palm_from_root.GetInverse() * M_flange_world)
    _weld_fixed(stage, flange, palm, f"{flange.GetPath()}/flange_to_robotiq")


def _attach_dynaarm(robot_prim_path: str, cfg: "AnymalDynaarmRobotiqSpawnCfg") -> None:
    import omni.physx.scripts.utils as physx_utils

    stage = get_current_stage()
    arm_prim_path = f"{robot_prim_path}/{cfg.arm_prim}"
    create_prim(
        arm_prim_path,
        usd_path=_arm_usd_path(),
        translation=cfg.mount_pos,
        orientation=cfg.mount_rot,
        stage=stage,
    )
    arm_root = stage.GetPrimAtPath(arm_prim_path)
    _strip_articulation_roots(arm_root)

    base_prim = stage.GetPrimAtPath(f"{robot_prim_path}/base")
    if not base_prim.IsValid():
        base_prim = _find_named_prim(stage.GetPrimAtPath(robot_prim_path), "base")
    arm_mount = _find_named_prim(arm_root, "arm_mount")
    if not base_prim.IsValid():
        raise RuntimeError(f"ANYmal-D 'base' prim not found under '{robot_prim_path}'.")
    if not arm_mount.IsValid():
        raise RuntimeError(f"DynaArm 'arm_mount' prim not found under '{arm_prim_path}'.")
    physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=base_prim, to_prim=arm_mount)
    _attach_isaac_robotiq(robot_prim_path, arm_root)


def _fix_link_to_world(prim) -> None:
    """Weld ``prim`` to the world with a FixedJoint (body0 omitted)."""
    from pxr import UsdPhysics

    stage = prim.GetStage()
    joint = UsdPhysics.FixedJoint.Define(stage, f"{prim.GetPath()}/WorldFixedJoint")
    joint.GetBody1Rel().SetTargets([prim.GetPath()])


@clone
def spawn_robotiq_2f140(
    prim_path: str,
    cfg: "Robotiq2F140SpawnCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn Isaac Lab's standalone Robotiq 2F-140, fixed in world."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    _strip_physics_scenes(prim)
    palm = _gripper_base_prim(prim)
    if not palm.IsValid():
        raise RuntimeError(
            f"Isaac Lab 2F-140 palm not found. Looked for {ROBOTIQ_GRIPPER_ROOT_ALIASES} under '{prim_path}'."
        )
    _fix_link_to_world(palm)
    return prim


@configclass
class Robotiq2F140SpawnCfg(sim_utils.UsdFileCfg):
    """Isaac Lab Nucleus 2F-140 (UR10e ``Robotiq_2f_140`` variant payload)."""

    func = spawn_robotiq_2f140
    usd_path: str = ISAAC_ROBOTIQ_2F140_USD


@clone
def spawn_anymal_d_dynaarm_robotiq(
    prim_path: str,
    cfg: "AnymalDynaarmRobotiqSpawnCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn ANYmal-D from USD, weld the DynaArm onto ``base``, then the 2F-140."""
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    _attach_dynaarm(prim_path, cfg)
    return prim


@configclass
class AnymalDynaarmRobotiqSpawnCfg(sim_utils.UsdFileCfg):
    """ANYmal-D USD plus DynaArm URDF plus Isaac Lab's 2F-140 USD."""

    func = spawn_anymal_d_dynaarm_robotiq
    arm_prim: str = ARM_PRIM
    mount_pos: tuple[float, float, float] = DYNAARM_MOUNT_POS
    mount_rot: tuple[float, float, float, float] = DYNAARM_MOUNT_ROT


_anymal_spawn = ANYMAL_D_CFG.spawn

ANYMAL_D_DYNAARM_ROBOTIQ_CFG = ANYMAL_D_CFG.replace(
    spawn=AnymalDynaarmRobotiqSpawnCfg(
        usd_path=_anymal_spawn.usd_path,
        activate_contact_sensors=_anymal_spawn.activate_contact_sensors,
        rigid_props=_anymal_spawn.rigid_props,
        articulation_props=_anymal_spawn.articulation_props.replace(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        collision_props=_anymal_spawn.collision_props,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,
            ".*F_HFE": 0.4,
            ".*H_HFE": -0.4,
            ".*F_KFE": -0.8,
            ".*H_KFE": 0.8,
            "dynaarm_shoulder_rotation": 0.0,
            "dynaarm_shoulder_flexion": -0.7,
            "dynaarm_elbow_flexion": 1.4,
            "dynaarm_forearm_rotation": 0.0,
            "dynaarm_wrist_flexion": 0.0,
            "dynaarm_wrist_rotation": 0.0,
            **ROBOTIQ_INIT_JOINT_POS,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ANYMAL_D_CFG.actuators["legs"],
        "dynaarm": ImplicitActuatorCfg(
            joint_names_expr=DYNAARM_JOINT_NAMES,
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper_drive": ROBOTIQ_DRIVE_ACTUATOR,
        "gripper_finger": ROBOTIQ_FINGER_ACTUATOR,
        "gripper_passive": ROBOTIQ_PASSIVE_ACTUATOR,
    },
    soft_joint_pos_limit_factor=0.95,
)
"""ANYmal-D USD with DynaArm corydoras12 and Isaac Lab's Robotiq 2F-140."""

ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG = ANYMAL_D_DYNAARM_ROBOTIQ_CFG.copy()
ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.actuators = dict(ANYMAL_D_DYNAARM_ROBOTIQ_CFG.actuators)
ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.actuators["dynaarm"] = ANYMAL_D_DYNAARM_ROBOTIQ_CFG.actuators["dynaarm"].replace(
    stiffness=200.0, damping=20.0
)
"""Same robot with stiffer DynaArm PD for differential IK."""

ROBOTIQ_2F140_CFG = ArticulationCfg(
    spawn=Robotiq2F140SpawnCfg(
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.15),
        joint_pos=dict(ROBOTIQ_INIT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "gripper_drive": ROBOTIQ_DRIVE_ACTUATOR,
        "gripper_finger": ROBOTIQ_FINGER_ACTUATOR,
        "gripper_passive": ROBOTIQ_PASSIVE_ACTUATOR,
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Standalone Isaac Lab Robotiq 2F-140 (UR10e ``Robotiq_2f_140`` payload)."""
