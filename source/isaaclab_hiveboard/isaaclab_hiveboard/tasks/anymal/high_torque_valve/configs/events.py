# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp

PI = math.pi


@configclass
class ValveEventCfg:
    """Configuration for events."""

    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    valve_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("high_torque_valve"), "gravcomp": 1.0},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.3, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    valve_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("high_torque_valve"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
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
    reset_valve_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.20, 0.20),
                "y": (-0.30, 0.30),
                "z": (-0.30, 0.30),
                "roll": (-PI / 6, PI / 6),
                "pitch": (-PI / 6, PI / 6),
                "yaw": (-PI / 5, PI / 5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("high_torque_valve"),
        },
    )
    reset_valve_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "high_torque_valve", joint_names=["RevoluteJoint", "PrismaticJoint"]
            ),
        },
    )


@configclass
class ValvePlayEventCfg(ValveEventCfg):
    """One fixed, reachable reset with no domain randomization."""

    robot_physics_material = None
    valve_physics_material = None

    def __post_init__(self):
        self.reset_valve_root.params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.reset_valve_joint.params["position_range"] = (0.0, 0.0)
