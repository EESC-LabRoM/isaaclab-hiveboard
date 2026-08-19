# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

from isaaclab_hiveboard.assets import BALL_VALVE_URDF, HONEYCOMB_USD


@configclass
class AnymalBallValveSceneCfg(InteractiveSceneCfg):
    """ANYmal C + HiveBoard Ball Valve scene."""

    # ANYmal robot articulation
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Warehouse backdrop
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        collision_group=-1,
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    # HiveBoard Panel
    panel = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Panel",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # HiveBoard Ball Valve Articulation
    ball_valve = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ball_valve",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=BALL_VALVE_URDF,
            fix_base=True,
            make_instanceable=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.1),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.5),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={"RevoluteJoint": 0.0},
        ),
        actuators={
            "valve_joint": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                effort_limit=50.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.1,
            )
        },
    )

    # Ground Plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
