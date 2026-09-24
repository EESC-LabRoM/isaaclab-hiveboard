# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard shock-absorber scene. Robot and EE frames are filled per robot.

Re-authored asset (see assets/hiveboard/shock_absorber), in the ``board`` link:

* plate x in [0, 0.015]; the spring (CoACD) runs up object +Z from its eye
  around the pin, reaching x = 0.036;
* pin on ``PrismaticJoint`` along +X at (y, z) = (0, 0.035), 0 = seated.
  Its 12 mm hex head (flats normal to object Z) spans
  x in [0.025 + q, 0.035 + q].

The pin starts 20 mm out, already through the eye; grasp and push it in,
then press the head down with closed fingers. The free pick-and-align stage
of the protocol is not modelled.
"""

from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import SHOCK_ABSORBER_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    FACE_QUAT,
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    face_roll_quat,
    honeycomb,
    mechanism_spawn,
    target_frames,
)

_ROOT = "{ENV_REGEX_NS}/ShockAbsorber/Geometry/board"
PIN_START = 0.020
PIN_SEATED = 0.004
PIN_AXIS_Z = 0.035
# Head top at reset.
PIN_HEAD_TOP_X = 0.035 + PIN_START
# 2F-140 pads reach ~2 cm past the mid-pad TCP; keep them above the spring eye.
PIN_GRASP_X = 0.058
FRANKA_PIN_TIP_X = PIN_HEAD_TOP_X - 0.008
# Push until the fingers would reach the spring eye, then press the rest.
PIN_PUSH = 0.012
# Closed-finger press target: 2F-140 fingertips ~2 cm past the TCP, 5 mm
# past the seated head top.
PIN_PRESS_X = 0.050
# Flats every 60 deg from object Z; 30 deg keeps the jaws off the spring.
PIN_PINCH_QUAT = face_roll_quat(30.0)

PIN_PINCH_FRAMES = ("pin_approaching", "pin_grasp", "pin_push")
PIN_GRASP_FRAMES = ("pin_grasp", "pin_push")


@configclass
class ShockAbsorberSceneCfg(InteractiveSceneCfg):
    """Shock absorber + honeycomb + canonical TCP frames."""

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    shock_absorber = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/ShockAbsorber",
        spawn=mechanism_spawn(SHOCK_ABSORBER_NEWTON_USD, "shock_absorber"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"PrismaticJoint": PIN_START},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "pin": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                # Loose press fit: sliding friction only [N].
                stiffness=0.0,
                damping=2.0,
                friction=0.5,
                effort_limit_sim=20.0,
            ),
        },
    )

    honeycomb = honeycomb(_ROOT, back_x=0.0)

    target_frame = target_frames(
        _ROOT,
        "ShockAbsorberTransformers",
        {
            "pin_approaching": OffsetCfg(pos=(0.15, 0.0, PIN_AXIS_Z), rot=PIN_PINCH_QUAT),
            "pin_grasp": OffsetCfg(pos=(PIN_GRASP_X, 0.0, PIN_AXIS_Z), rot=PIN_PINCH_QUAT),
            "pin_push": OffsetCfg(pos=(PIN_GRASP_X - PIN_PUSH, 0.0, PIN_AXIS_Z), rot=PIN_PINCH_QUAT),
            # Closed fingertips over the head, then past its seated top.
            "press_approaching": OffsetCfg(pos=(PIN_HEAD_TOP_X + 0.03, 0.0, PIN_AXIS_Z), rot=FACE_QUAT),
            "press": OffsetCfg(pos=(PIN_PRESS_X, 0.0, PIN_AXIS_Z), rot=FACE_QUAT),
        },
    )
