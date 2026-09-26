# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard sliding-drawer scene. Robot and EE frames are filled per robot.

Drawer_Assembly in its housing frame (see assets/hiveboard/drawer):

* housing: x in [0, 0.038], plate at the back (x = 0);
* box: x in [0.0027, 0.0477], z in [-0.022, 0.0216]; its front
  sticks ~10 mm out of the case.
* The box is a free rigid body. The housing is welded to the world. Shafts
  and side cuts are the only guide: forward along the slot, then up, out.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import DRAWER_BOX_NEWTON_USD, DRAWER_HOUSING_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    FACE_QUAT,
    honeycomb,
    target_frames,
)


def _body_props(*, kinematic: bool) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic,
        disable_gravity=True,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


def _housing_spawn() -> sim_utils.UsdFileCfg:
    """Welded case. The root joint in the USD holds it; contact cannot shove it."""
    return sim_utils.UsdFileCfg(
        usd_path=DRAWER_HOUSING_NEWTON_USD,
        activate_contact_sensors=True,
        rigid_props=_body_props(kinematic=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True),
        semantic_tags=[("class", "drawer_housing")],
    )


def _box_spawn() -> sim_utils.UsdFileCfg:
    """Free box. Nothing but the shaft-to-cut contact guides it."""
    return sim_utils.UsdFileCfg(
        usd_path=DRAWER_BOX_NEWTON_USD,
        activate_contact_sensors=True,
        rigid_props=_body_props(kinematic=False),
        semantic_tags=[("class", "drawer")],
    )


_HOUSING = "{ENV_REGEX_NS}/DrawerHousing/Geometry/base"
# Pinch the handle tab across its 30 mm width (object Y); at 5 mm it is too
# thin across Z for the 2F-140, which stops ~12 mm short of closing.
DRAWER_TAB_CENTER_YZ = (0.001, 0.009)
# TCP x for grippers whose TCP is mid-pad (2F-140, Spot): the ~4 cm pads then
# end just short of the drawer face at x = 0.038.
DRAWER_GRASP_X = 0.058
# Commanded pull, still inside the slot (the lip is further forward).
DRAWER_PULL = 0.023


@configclass
class DrawerSceneCfg(InteractiveSceneCfg):
    """Free drawer box, fixed housing, honeycomb and canonical TCP frames.

    robot and ee_frame are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    drawer_housing = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/DrawerHousing",
        spawn=_housing_spawn(),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    drawer = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Drawer",
        spawn=_box_spawn(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=MECHANISM_SPAWN_POS, rot=FACE_QUAT),
    )

    honeycomb = honeycomb(_HOUSING, back_x=0.0)

    # FACE_QUAT puts jaws that close across TCP +Y across the tab's width.
    # Frames stay on the housing, at the handle tab, not on the moving box.
    target_frame = target_frames(
        _HOUSING,
        "DrawerTransformers",
        {
            "drawer_approaching": OffsetCfg(pos=(0.13, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
            "drawer_grasp": OffsetCfg(pos=(DRAWER_GRASP_X, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
            "drawer_pulled": OffsetCfg(pos=(DRAWER_GRASP_X + DRAWER_PULL, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
        },
    )
