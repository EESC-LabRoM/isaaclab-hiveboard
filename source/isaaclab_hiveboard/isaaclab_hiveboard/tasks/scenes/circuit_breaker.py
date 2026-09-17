# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard circuit breaker scene. Robot and EE frames are filled per robot."""

from dataclasses import MISSING

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import CIRCUIT_BREAKER_USD, HONEYCOMB_USD


@configclass
class CircuitBreakerSceneCfg(InteractiveSceneCfg):
    """Circuit breaker + honeycomb + canonical TCP frames.

    ``robot`` and ``ee_frame`` are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    circuit_breaker = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CIRCUIT_BREAKER_USD,
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            semantic_tags=[("class", "circuit_breaker")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Old PhysX scene spawned the breaker at (1.0, 0.0, 0.0) with Spot
            # at z=0. Newton Spot stands at z=0.65, so lift the panel to the
            # same height relative to the body. 180 deg yaw faces Spot.
            pos=(1.0, 0.0, 0.65),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "joint_actuator": ImplicitActuatorCfg(
                damping=0.0,
                friction=0.02,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit_sim=2.0,
                joint_names_expr=["RevoluteJoint"],
                # Passive switch. A non-zero drive would pull every custom
                # reset angle toward its stale target on the first step.
                stiffness=0.0,
            ),
        },
    )

    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.04, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        collision_group=-1,
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CircuitBreakerTransformers"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World",
                name="approaching",
                offset=OffsetCfg(
                    pos=(0.10, 0.02, 0.0),
                    # 180 deg yaw about Z (xyzw). Was wxyz (0,0,0,1).
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World",
                name="lever_pivot_below",
                offset=OffsetCfg(
                    pos=(0.04, 0.0, -0.07),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World",
                name="lever_pivot_above",
                offset=OffsetCfg(
                    pos=(0.04, 0.0, 0.07),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/Geometry/World",
                name="rotate_frame",
                offset=OffsetCfg(
                    pos=(0.02, 0.0, 0.0),
                    rot=(0.0, 0.0, 1.0, 0.0),
                ),
            ),
        ],
    )
