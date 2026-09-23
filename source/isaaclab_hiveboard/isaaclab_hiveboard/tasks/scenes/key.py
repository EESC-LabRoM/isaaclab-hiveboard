# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard lock-and-key scene. Robot and EE frames are filled per robot.

Key_Assembly layout in its lock link frame (articulated re-authoring of the
rigid upstream asset, see assets/hiveboard/key):

* lock plate: x in [-0.010, 0.005];
* key: fully inserted, shaft along +X; its bow is a 24 x 24 mm ring, 7 mm
  thick (object Y), spanning x in [0.049, 0.073], z in [-0.013, 0.011].
  RevoluteJoint about +X in [0, 90] deg.
"""

from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import KEY_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    FACE_QUAT,
    honeycomb,
    mechanism_spawn,
    target_frames,
)

_ROOT = "{ENV_REGEX_NS}/Key/Geometry/lock"
# Pinch the 24 x 24 mm bow ring across its ~34 mm diagonal, on the key axis
# so the turn is a pure wrist roll. The simulated 2F-140 bottoms out ~24 mm
# apart, so it cannot grip the ring's 7 mm thickness or, reliably, its 24 mm
# height. ``KEY_GRASP_X`` is for grippers whose TCP is mid-pad (2F-140,
# Spot): the ~4 cm pads then reach the bow's inner edge (x = 0.049) without
# touching the lock plate.
KEY_GRASP_X = 0.068
KEY_AXIS_Z = -0.001
# FACE_QUAT rolled 45 deg about TCP +X: jaws across the ring's diagonal.
KEY_PINCH_QUAT = (0.0, 0.38268343, 0.92387953, 0.0)

@configclass
class KeySceneCfg(InteractiveSceneCfg):
    """Lock and key + honeycomb + canonical TCP frames.

    robot and ee_frame are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    key = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Key",
        spawn=mechanism_spawn(KEY_NEWTON_USD, "key"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"RevoluteJoint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "lock": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                # Passive cylinder with a little turning friction [N m].
                stiffness=0.0,
                damping=0.01,
                friction=0.02,
                effort_limit_sim=2.0,
            ),
        },
    )

    honeycomb = honeycomb(_ROOT, back_x=-0.010)

    target_frame = target_frames(
        _ROOT,
        "KeyTransformers",
        {
            "key_approaching": OffsetCfg(pos=(0.15, 0.0, KEY_AXIS_Z), rot=KEY_PINCH_QUAT),
            "key_grasp": OffsetCfg(pos=(KEY_GRASP_X, 0.0, KEY_AXIS_Z), rot=KEY_PINCH_QUAT),
            # On the key axis; the turn is about object +X (frame -X).
            "rotate_frame": OffsetCfg(pos=(KEY_GRASP_X, 0.0, KEY_AXIS_Z), rot=FACE_QUAT),
        },
    )
