# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HiveBoard peg-insertion scene. Robot and EE frames are filled per robot.

Re-authored asset (see assets/hiveboard/peg_insertion), in the root link:
plate x in [0, 0.015] with five holes; the threaded peg sits in the centre
hole and spans x in [q, 0.038 + q] for PrismaticJoint q in [0, 0.020]. Its
shank is 7 mm across up to the thread, 5 mm above it.

The peg starts engaged two turns out; the free pick-and-align stage of the
protocol is not modelled.
"""

from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import PEG_INSERTION_NEWTON_USD
from isaaclab_hiveboard.tasks.scenes.mechanism import GROUND, LIGHT, honeycomb
from isaaclab_hiveboard.tasks.scenes.screw import ScrewSpec, screw_articulation, screw_frames

PEG_SPEC = ScrewSpec(
    pitch=0.00125,
    start=0.0025,
    travel=0.020,
    grasp_x=0.037,
    franka_tip_x=0.022,
    jaw_roll_deg=0.0,
    seated_tolerance=0.0005,
    friction=0.02,
)

_ROOT = "{ENV_REGEX_NS}/Peg/Geometry/World"


@configclass
class PegInsertionSceneCfg(InteractiveSceneCfg):
    """Peg plate + threaded peg + honeycomb + canonical TCP frames."""

    robot: ArticulationCfg = MISSING  # type: ignore
    ee_frame: FrameTransformerCfg = MISSING  # type: ignore
    ground = GROUND
    light = LIGHT
    peg = screw_articulation("Peg", PEG_INSERTION_NEWTON_USD, "peg_insertion", PEG_SPEC)
    honeycomb = honeycomb(_ROOT, back_x=0.0)
    target_frame = screw_frames(_ROOT, "PegTransformers", PEG_SPEC)
