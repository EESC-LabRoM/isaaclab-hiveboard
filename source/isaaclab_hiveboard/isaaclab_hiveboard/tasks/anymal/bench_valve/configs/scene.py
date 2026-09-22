# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upright HiveBoard scene matching the Spot bench demo, robot swapped for ANYmal."""

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils

from isaaclab_hiveboard.assets import BALL_VALVE_USD, HONEYCOMB_USD
from isaaclab_hiveboard.assets.anymal.anymal import ROBOTIQ_LEFT_PAD_PRIM, ROBOTIQ_RIGHT_PAD_PRIM
from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_NEWTON_CFG,
    BOARD_POS,
    BOARD_QUAT_XYZW,
    STAND_FOOT_POS,
    STAND_FOOT_SIZE,
    STAND_POST_POS,
    STAND_POST_SIZE,
    VALVE_ARMATURE,
    VALVE_DAMPING,
    VALVE_FRICTION,
    VALVE_POS,
    VALVE_QUAT_XYZW,
)

_VALVE_CONTACT_FILTER = ["{ENV_REGEX_NS}/Valve/.*"]


@configclass
class BenchValveSceneCfg(InteractiveSceneCfg):
    """Fixed-base ANYmal + DynaArm, upright honeycomb, and ball valve at bench poses."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

    stand_post = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StandPost",
        spawn=sim_utils.CuboidCfg(
            size=STAND_POST_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=STAND_POST_POS),
        collision_group=-1,
    )

    stand_foot = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/StandFoot",
        spawn=sim_utils.CuboidCfg(
            size=STAND_FOOT_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=STAND_FOOT_POS),
        collision_group=-1,
    )

    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=BOARD_POS, rot=BOARD_QUAT_XYZW),
        collision_group=-1,
    )

    ball_valve = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BALL_VALVE_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
            semantic_tags=[("class", "valve")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VALVE_POS,
            rot=VALVE_QUAT_XYZW,
            joint_pos={"RevoluteJoint": 0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "joint_actuator": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.01,
                damping=VALVE_DAMPING,
                friction=VALVE_FRICTION,
                armature=VALVE_ARMATURE,
                effort_limit_sim=100.0,
            ),
        },
    )

    finger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_LEFT_PAD_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_VALVE_CONTACT_FILTER,
    )
    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_RIGHT_PAD_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_VALVE_CONTACT_FILTER,
    )

    scene_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam",
        offset=CameraCfg.OffsetCfg(
            pos=(0.2, -2.1, 0.9),
            rot=(-0.02862, 0.03624, 0.61917, 0.78390),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=1920,
        height=1080,
        update_period=0.0,
        renderer_cfg=NewtonWarpRendererCfg(enable_shadows=True),
    )
