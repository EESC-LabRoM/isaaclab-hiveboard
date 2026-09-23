# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard sliding-drawer scene. Robot and EE frames are filled per robot.

Drawer_Assembly layout in its base link frame (articulated re-authoring of
the rigid upstream asset, see assets/hiveboard/drawer):

* base box: x in [0, 0.038], plate at the back (x = 0);
* drawer: x in [0.0027, 0.0477], z in [-0.022, 0.0216]; its front
  sticks ~10 mm out of the box. PrismaticJoint along +X in [0, 0.025].
"""

from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import DRAWER_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    FACE_QUAT,
    honeycomb,
    mechanism_spawn,
    target_frames,
)

_ROOT = "{ENV_REGEX_NS}/Drawer/Geometry/base"
# Pinch the handle tab across its 30 mm width (object Y); at 5 mm it is too
# thin across Z for the 2F-140, which stops ~12 mm short of closing.
DRAWER_TAB_CENTER_YZ = (0.001, 0.009)
# TCP x for grippers whose TCP is mid-pad (2F-140, Spot): the ~4 cm pads then
# end just short of the drawer face at x = 0.038.
DRAWER_GRASP_X = 0.058
# Pull a little short of the 25 mm stop.
DRAWER_PULL = 0.023


@configclass
class DrawerSceneCfg(InteractiveSceneCfg):
    """Sliding drawer + honeycomb + canonical TCP frames.

    robot and ee_frame are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    drawer = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drawer",
        spawn=mechanism_spawn(DRAWER_NEWTON_USD, "drawer"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"PrismaticJoint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "slide": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                # Passive slide with a little sliding friction [N].
                stiffness=0.0,
                damping=1.0,
                friction=0.2,
                effort_limit_sim=20.0,
            ),
        },
    )

    honeycomb = honeycomb(_ROOT, back_x=0.0)

    # FACE_QUAT puts jaws that close across TCP +Y across the tab's width.
    target_frame = target_frames(
        _ROOT,
        "DrawerTransformers",
        {
            "drawer_approaching": OffsetCfg(pos=(0.13, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
            "drawer_grasp": OffsetCfg(pos=(DRAWER_GRASP_X, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
            "drawer_pulled": OffsetCfg(pos=(DRAWER_GRASP_X + DRAWER_PULL, *DRAWER_TAB_CENTER_YZ), rot=FACE_QUAT),
        },
    )
