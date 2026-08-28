# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard hidden-button scene. Robot and EE frames are filled per robot."""

from dataclasses import MISSING

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_hiveboard.assets import BUTTON_URDF, HONEYCOMB_USD, SPOT_WORKSPACE

# URDF default (0) is already mostly open. Closed is ~-90 deg (cover on the
# button). lid_push is the into-face / +Y pose that previously loaded the
# cover; RotateFrame about lid_hinge +Z then continues that hinge arc.
_LID_CLOSED_POS = -1.56
_APPROACH_POS = (0.120, 0.000, 0.0)
_LID_PRE_CONTACT_POS = (0.020, -0.07, 0.0)
_LID_CONTACT_POS = (0.010, -0.05, 0.0)
# After the hinge arc, raise off the cover first, then back toward Spot
# at that height, then down to the button line.
_LID_LIFT_POS = (0.070, 0.060, 0.040)
_LID_CLEAR_POS = (0.200, 0.00, 0.040)
_LID_SAFE_POS = (0.200, 0.000, 0.000)
_FRAME_ROT = (0.0, 0.0, 0.0, 1.0)
_LID_PIVOT_PATH = "{ENV_REGEX_NS}/Button/lid_pivot"


@configclass
class ButtonSceneCfg(InteractiveSceneCfg):
    """Hidden button + honeycomb + canonical TCP frames.

    ``robot`` and ``ee_frame`` are filled by the robot-specific subclass.
    Default placement matches Spot.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore

    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SPOT_WORKSPACE.warehouse_pos),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    button = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=BUTTON_URDF,
            activate_contact_sensors=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None),  # type: ignore
            ),
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
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            semantic_tags=[("class", "button")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=SPOT_WORKSPACE.object_pos,
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "RevoluteJoint": _LID_CLOSED_POS,
                "PrismaticJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "lid_actuator": ImplicitActuatorCfg(
                damping=0.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=2.0,
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.0,
            ),
            "button_spring": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                stiffness=10.0,
                damping=4.0,
                effort_limit=50.0,
            ),
        },
    )

    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Button/World/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.04, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        collision_group=-1,
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Button/World",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ButtonTransformers"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="approaching",
                offset=OffsetCfg(
                    pos=_APPROACH_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="lid_contact",
                offset=OffsetCfg(
                    pos=_LID_PRE_CONTACT_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="lid_contact",
                offset=OffsetCfg(
                    pos=_LID_CONTACT_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=_LID_PIVOT_PATH,
                name="lid_hinge",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="lid_lift",
                offset=OffsetCfg(
                    pos=_LID_LIFT_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="lid_clear",
                offset=OffsetCfg(
                    pos=_LID_CLEAR_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="lid_safe",
                offset=OffsetCfg(
                    pos=_LID_SAFE_POS,
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="button_approach",
                offset=OffsetCfg(
                    pos=(0.08, 0.0, 0.0),
                    rot=_FRAME_ROT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="button_press",
                offset=OffsetCfg(
                    pos=(0.01, 0.0, 0.0),
                    rot=_FRAME_ROT,
                ),
            ),
        ],
    )
