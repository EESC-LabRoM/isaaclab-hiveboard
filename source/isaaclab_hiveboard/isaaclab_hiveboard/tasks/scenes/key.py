# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard lock-and-key scene: the robot carries the key in, inserts and turns it.

Robot and EE frames are filled per robot; the robot subclass also glues the
key in its hand with :func:`key_in_hand`. The lock and the key are separate
assets (see assets/hiveboard/key), laid out in the lock link frame:

* lock housing: x in [-0.010, 0.005], front face at x = 0.005;
* plug: a keyway open toward +X, closed by a back stop at x = -0.0065.
  RevoluteJoint about +X in [0, 90] deg; at 0 the keyway lines up with the
  visual keyhole;
* key: its link frame is its pose when fully inserted at RevoluteJoint = 0
  (blade tip at x = -0.006). The bow spans x in [0.048, 0.073], 24 mm across
  lock Y and 7 mm thick in lock Z.

Inserted, the blade sits in the keyway with 1.2 mm of clearance per side, so
the turn drives the plug after about 5 deg of backlash.
"""

from dataclasses import MISSING

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import KEY_NEWTON_USD, LOCK_NEWTON_USD, EndEffectorCfg
from isaaclab_hiveboard.assets.glued_tool import GluedToolCfg, glue_tool, quat_mul, roll_about_x, tool_pose_in_link
from isaaclab_hiveboard.tasks.scenes.mechanism import (
    GROUND,
    LIGHT,
    MECHANISM_SPAWN_POS,
    FACE_QUAT,
    honeycomb,
    mechanism_spawn,
    target_frames,
)

_ROOT = "{ENV_REGEX_NS}/Lock/Geometry/lock"
# TCP along the key for grippers whose TCP is mid-pad (2F-140, Spot): the
# ~4 cm pads then cover the bow (x in [0.048, 0.073]) and stay clear of the
# lock face once the key is in.
KEY_GRASP_X = 0.068
# Key backed out along the lock axis before the insertion [m]: approach with
# the blade tip 7.4 cm off the face, then enter with it 7 mm off.
KEY_APPROACH_X = 0.080
KEY_ENTRY_X = 0.012


def key_tcp_quat(jaw_roll_deg: float = 0.0) -> tuple[float, float, float, float]:
    """TCP orientation in the key (and lock) frame: FACE_QUAT rolled ``jaw_roll_deg`` about the TCP's own +X.

    At 0, jaws that close across canonical TCP +Y close across the bow's
    24 mm width; at 90 they pinch its 7 mm thickness. Each robot picks the
    roll that keeps its wrist inside its limits through the quarter turn.
    """
    return quat_mul(FACE_QUAT, roll_about_x(jaw_roll_deg))


def key_in_hand(
    robot: ArticulationCfg,
    ee: EndEffectorCfg,
    grasp_x: float = KEY_GRASP_X,
    jaw_roll_deg: float = 0.0,
) -> ArticulationCfg:
    """``robot`` with the key welded to its hand, the TCP at ``grasp_x`` along the key.

    Must match the :func:`key_frames` the task targets with.
    """
    pos, rot = tool_pose_in_link(ee, ((grasp_x, 0.0, 0.0), key_tcp_quat(jaw_roll_deg)))
    tool = GluedToolCfg(usd_path=KEY_NEWTON_USD, body_name=ee.body_name, prim_name="Key", pos=pos, rot=rot)
    return glue_tool(robot, tool)


def key_frames(grasp_x: float = KEY_GRASP_X, jaw_roll_deg: float = 0.0) -> FrameTransformerCfg:
    """TCP goals for a key glued by :func:`key_in_hand` with the same arguments."""
    rot = key_tcp_quat(jaw_roll_deg)
    return target_frames(
        _ROOT,
        "KeyTransformers",
        {
            "key_approaching": OffsetCfg(pos=(grasp_x + KEY_APPROACH_X, 0.0, 0.0), rot=rot),
            "key_entry": OffsetCfg(pos=(grasp_x + KEY_ENTRY_X, 0.0, 0.0), rot=rot),
            "key_inserted": OffsetCfg(pos=(grasp_x, 0.0, 0.0), rot=rot),
            # On the key axis; the turn is about object +X (frame -X).
            "rotate_frame": OffsetCfg(pos=(grasp_x, 0.0, 0.0), rot=FACE_QUAT),
        },
    )


@configclass
class KeySceneCfg(InteractiveSceneCfg):
    """Lock + honeycomb + canonical TCP frames.

    robot (with :func:`key_in_hand`) and ee_frame are filled by the robot-specific subclass.
    """

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT

    lock = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Lock",
        spawn=mechanism_spawn(LOCK_NEWTON_USD, "lock"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"RevoluteJoint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "plug": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                # Passive plug with a little turning friction [N m].
                stiffness=0.0,
                damping=0.01,
                friction=0.02,
                effort_limit_sim=2.0,
                # MJWarp contacts are mass-normalized: on the bare 2e-6 kg m^2
                # plug the blade sank into the keyway walls and led the plug by
                # up to 37 deg. Rotor inertia makes the walls hold.
                armature=1.0e-3,
            ),
        },
    )

    honeycomb = honeycomb(_ROOT, back_x=-0.010)

    target_frame = key_frames()
