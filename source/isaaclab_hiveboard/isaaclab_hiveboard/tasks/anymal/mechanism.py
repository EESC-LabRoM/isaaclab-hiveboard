# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP pieces for the small HiveBoard mechanisms (button, drawer, key).

These tasks differ only in the object entity and the joints that make up its
mechanism, so the physics preset, events and observations are built here from
``(asset_name, joint_names)`` instead of being copied per task.
"""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard import mdp as spot_mdp
from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.anymal.anymal import ROBOTIQ_LEFT_PAD_PRIM, ROBOTIQ_RIGHT_PAD_PRIM
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES
from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp

PI = math.pi

ANYMAL_CUROBO = {
    "robot_joint_names": list(ANYMAL_ARM_JOINT_NAMES),
    "robot_curobo_yaml": f"{ASSET_DIR}/anymal/cumotion/dynaarm.yaml",
    "robot_urdf": f"{ASSET_DIR}/anymal/cumotion/dynaarm.urdf",
}


def pad_contacts(object_prim: str) -> tuple[ContactSensorCfg, ContactSensorCfg]:
    """2F-140 left/right pad contact sensors filtered to ``object_prim``'s subtree."""
    return tuple(
        ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + pad,
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{object_prim}/.*"],
        )
        for pad in (ROBOTIQ_LEFT_PAD_PRIM, ROBOTIQ_RIGHT_PAD_PRIM)
    )


# Per-world contact budget. Newton auto-sizes its contact buffer for all
# worlds from the shape count, and MJWarp refuses to start when
# nconmax x num_envs exceeds it. These mechanisms have only a handful of
# shapes (FR3 + button estimates 1775), so the valves' 4000 is too many.
_NCONMAX = 1000
_NO_POSE_NOISE = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}


@configclass
class MechanismPhysicsCfg(PresetCfg):
    """Physics variants exposed through ``physics=...`` on the Isaac Lab 3 CLI."""

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=_NCONMAX,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=False,
    )
    default = newton_mjwarp
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        friction_correlation_distance=0.00625,
    )


@configclass
class MechanismEventCfg:
    """Gravity compensation, materials, joint friction and root randomization.

    Build with :func:`mechanism_events`; the ``"object"`` entity is a placeholder.
    """

    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    object_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "gravcomp": 1.0},
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    object_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "friction_distribution_params": (0.005, 0.05),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_object_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # These mechanisms are only a few centimeters across, so keep
            # placement noise moderate.
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.15, 0.15),
                "z": (-0.15, 0.15),
                "roll": (-PI / 18, PI / 18),
                "pitch": (-PI / 18, PI / 18),
                "yaw": (-PI / 12, PI / 12),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


def mechanism_events(asset_name: str, friction_joints: list[str], play: bool = False) -> MechanismEventCfg:
    """Events for ``asset_name``; ``play`` drops randomization for a fixed demo reset."""
    events = MechanismEventCfg()
    events.object_gravcomp.params["asset_cfg"] = SceneEntityCfg(asset_name)
    events.object_joint_parameters.params["asset_cfg"] = SceneEntityCfg(asset_name, joint_names=friction_joints)
    events.object_physics_material.params["asset_cfg"] = SceneEntityCfg(asset_name)
    events.reset_object_root.params["asset_cfg"] = SceneEntityCfg(asset_name)
    if play:
        events.robot_physics_material = None
        events.object_joint_parameters = None
        events.object_physics_material = None
        events.reset_object_root.params["pose_range"] = dict(_NO_POSE_NOISE)
    return events


_ARM = SceneEntityCfg("robot", joint_names=ANYMAL_ARM_JOINT_NAMES, preserve_order=True)
_EE_POSE = ObsTerm(func=spot_mdp.ee_pose_b, params={"asset_cfg": SceneEntityCfg("robot"), "frame_name": "ee_frame"})


@configclass
class MechanismObservationsCfg:
    """Observation groups. Build with :func:`mechanism_observations`."""

    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        finger_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w, params={"sensor_cfg": SceneEntityCfg("finger_contact")}
        )
        jaw_contact = ObsTerm(func=spot_mdp.contact_net_forces_w, params={"sensor_cfg": SceneEntityCfg("jaw_contact")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class DiffusionPolicyCfg(ObsGroup):
        ee_pose_b = _EE_POSE
        arm_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": _ARM})
        arm_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": _ARM})
        object_root_pose_b = ObsTerm(
            func=spot_mdp.object_root_pose_b,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
        )
        object_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("object")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class EvaluationCfg(ObsGroup):
        """Named physical signals used by policy evaluation metrics."""

        ee_pose_b = _EE_POSE
        object_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("object")})
        finger_contact = ObsTerm(
            func=spot_mdp.contact_net_forces_w, params={"sensor_cfg": SceneEntityCfg("finger_contact")}
        )
        jaw_contact = ObsTerm(func=spot_mdp.contact_net_forces_w, params={"sensor_cfg": SceneEntityCfg("jaw_contact")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    diffusion_policy: DiffusionPolicyCfg = DiffusionPolicyCfg()
    evaluation: EvaluationCfg = EvaluationCfg()


def mechanism_observations(asset_name: str, joint_names: list[str]) -> MechanismObservationsCfg:
    """Observations for ``asset_name`` with its mechanism ``joint_names`` in order."""
    observations = MechanismObservationsCfg()
    joints = SceneEntityCfg(asset_name, joint_names=joint_names, preserve_order=True)
    observations.diffusion_policy.object_root_pose_b.params["object_cfg"] = SceneEntityCfg(asset_name)
    observations.diffusion_policy.object_joint_pos.params["asset_cfg"] = joints
    observations.evaluation.object_joint_pos.params["asset_cfg"] = joints
    return observations
