# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard M8 and M30 thread scenes. Robot and EE frames are filled per robot.

Re-authored assets (see assets/hiveboard/{m8,m30}_thread), in the root link:

* M8: plate x in [-0.025, -0.010], collar to x = 0, M8 rod to x = 0.020.
  Nut 13 mm across flats (flats normal to object Z), 6 mm thick, spans
  x in [q, q + 0.006] for PrismaticJoint q in [0, 0.015].
* M30: plate x in [0, 0.015], M30 rod to x = 0.065. Nut 46 mm across flats,
  24 mm thick, spans x in [0.0156 + q, 0.0396 + q] for q in [0, 0.025].

Both start two turns out from seated; the protocol's success is seating it.
"""

from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import M8_THREAD_NEWTON_USD, M30_THREAD_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import GROUND, LIGHT, honeycomb
from isaaclab_hiveboard.tasks.scenes.screw import ScrewSpec, screw_articulation, screw_frames

M8_SPEC = ScrewSpec(
    pitch=0.00125,
    start=0.0025,
    travel=0.015,
    # 2F-140 pads reach ~2 cm past the TCP; keep them off the plate.
    grasp_x=0.012,
    # The FR3 fingertips reach a few mm past its TCP: at -0.005 they drag on
    # the plate, at 0.0 the pads barely overlap the 6 mm nut.
    franka_tip_x=-0.002,
    # Flats every 60 deg from object Z.
    jaw_roll_deg=30.0,
    seated_tolerance=0.0005,
    friction=0.02,
)
M30_SPEC = ScrewSpec(
    pitch=0.0035,
    start=0.007,
    travel=0.025,
    grasp_x=0.036,
    # The rod runs 18 mm past the nut; keep the FR3 palm clear of its end.
    franka_tip_x=0.030,
    # The nut mesh is yawed -25.4 deg about the thread, putting a flat
    # normal 4.6 deg off object -Y.
    jaw_roll_deg=-4.6,
    seated_tolerance=0.001,
    friction=0.1,
)

_M8_ROOT = "{ENV_REGEX_NS}/Thread/Geometry/World"
_M30_ROOT = _M8_ROOT


@configclass
class M8SceneCfg(InteractiveSceneCfg):
    """M8 screw and nut + honeycomb + canonical TCP frames."""

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT
    thread = screw_articulation("Thread", M8_THREAD_NEWTON_USD, "m8_thread", M8_SPEC)
    honeycomb = honeycomb(_M8_ROOT, back_x=-0.0254)
    target_frame = screw_frames(_M8_ROOT, "M8Transformers", M8_SPEC)


@configclass
class M30SceneCfg(InteractiveSceneCfg):
    """M30 thread and nut + honeycomb + canonical TCP frames."""

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT
    thread = screw_articulation("Thread", M30_THREAD_NEWTON_USD, "m30_thread", M30_SPEC)
    honeycomb = honeycomb(_M30_ROOT, back_x=0.0)
    target_frame = screw_frames(_M30_ROOT, "M30Transformers", M30_SPEC)
