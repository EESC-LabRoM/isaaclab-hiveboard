# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pieces shared by the small HiveBoard mechanism scenes (button, drawer, key)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

from isaaclab_hiveboard.assets import HONEYCOMB_USD

# Same placement as the circuit breaker: 1 m ahead of the legged robots'
# shoulders, 0.65 m up, with a 180 deg yaw so the mechanism faces the robot.
MECHANISM_SPAWN_POS = (1.0, 0.0, 0.65)
# 180 deg yaw (xyzw): canonical TCP +X points into the plate (object -X).
FACE_QUAT = (0.0, 0.0, 1.0, 0.0)
# FACE_QUAT then a 90 deg roll about TCP +X: jaws that close across TCP +Y
# close across object Z.
PINCH_Z_QUAT = (0.0, 0.70710678, 0.70710678, 0.0)

GROUND = AssetBaseCfg(
    prim_path="/World/Ground",
    spawn=sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    collision_group=-1,
)

LIGHT = AssetBaseCfg(
    prim_path="/World/light",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
)


def mechanism_spawn(usd_path: str, semantic_class: str) -> sim_utils.UsdFileCfg:
    """Fixed, gravity-free articulation spawn with contact reporting."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        semantic_tags=[("class", semantic_class)],
    )


def honeycomb(root_link_prim: str, back_x: float) -> AssetBaseCfg:
    """Visual-only honeycomb panel whose 20 mm face ends flush at back_x."""
    return AssetBaseCfg(
        prim_path=f"{root_link_prim}/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(back_x - 0.020, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        collision_group=-1,
    )


def target_frames(root_link_prim: str, marker: str, frames: dict[str, OffsetCfg]) -> FrameTransformerCfg:
    """Named canonical TCP goals, all expressed in the fixed root link."""
    return FrameTransformerCfg(
        prim_path=root_link_prim,
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path=f"/Visuals/{marker}"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path=root_link_prim, name=name, offset=offset)
            for name, offset in frames.items()
        ],
    )
