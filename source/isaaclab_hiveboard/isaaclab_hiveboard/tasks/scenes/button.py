# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard hidden push-button scene. Robot and EE frames are filled per robot.

Button_Assembly layout in its ``World`` link frame (from the UUC asset):

* base plate: ``x in [-0.021, 0]``, front face at ``x = 0``;
* button: protrudes to ``x = 0.015`` (radius 22.5 mm), presses along -X,
  ``PrismaticJoint`` in ``[-0.01, 0]``;
* lid: ``RevoluteJoint`` about +Z through the hinge at ``y = 0.0305``, limits
  ``[-90, 40]`` deg. At -90 deg it covers the button face; at 0 it stands
  straight out of the plate.
"""

import math
from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import BUTTON_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    FACE_QUAT,
    PINCH_Z_QUAT,
    honeycomb,
    mechanism_spawn,
    target_frames,
)

_ROOT = "{ENV_REGEX_NS}/Button/Geometry/World"
LID_CLOSED = -math.pi / 2
LID_HINGE_POS = (-0.0034325, 0.0304632, 0.0)
# The jaws close across the lid's 56 mm width (object Z); its thickness sits
# flat on the plate and cannot be pinched.
LID_PINCH_QUAT = PINCH_Z_QUAT
# Pinch the closed lid 13 mm in from its free edge (object y = -0.031).
LID_PINCH_Y = -0.018


@configclass
class ButtonSceneCfg(InteractiveSceneCfg):
    """Hidden push button + honeycomb + canonical TCP frames.

    ``robot`` and ``ee_frame`` are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    button = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=mechanism_spawn(BUTTON_NEWTON_USD, "button"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"RevoluteJoint": LID_CLOSED, "PrismaticJoint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "lid": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                # Passive hinge; a drive would pull the lid back to its reset target.
                stiffness=0.0,
                damping=0.0,
                friction=0.02,
                effort_limit_sim=2.0,
            ),
            "button_spring": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                # Return spring: ~1 N at the full 10 mm stroke.
                stiffness=100.0,
                damping=2.0,
                effort_limit_sim=5.0,
            ),
        },
    )

    # The 21 mm plate's back face is at x = -0.021.
    honeycomb = honeycomb(_ROOT, back_x=-0.021)

    target_frame = target_frames(
        _ROOT,
        "ButtonTransformers",
        {
            "lid_approaching": OffsetCfg(pos=(0.10, LID_PINCH_Y, 0.0), rot=LID_PINCH_QUAT),
            "lid_grasp": OffsetCfg(pos=(0.006, LID_PINCH_Y, 0.0), rot=LID_PINCH_QUAT),
            "rotate_frame": OffsetCfg(pos=LID_HINGE_POS, rot=FACE_QUAT),
            "button_approaching": OffsetCfg(pos=(0.10, 0.0, 0.0), rot=FACE_QUAT),
            # Button face is at x = 0.015 and bottoms out at 0.005; aim past
            # that so the press holds against the stop.
            "button_press": OffsetCfg(pos=(0.002, 0.0, 0.0), rot=FACE_QUAT),
        },
    )
