# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, ANYMAL_WORKSPACE, make_ee_frame
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LampSceneCfg


@configclass
class AnymalLampSceneCfg(LampSceneCfg):
    """ANYmal-D + DynaArm + 2F-140 facing the shared screw-in lamp."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    # The shared lamp scene defines Spot-only contact sensors on the arm/jaw
    # links; ANYmal's 2F-140 pad contacts aren't wired into this task.
    finger_contact = None
    jaw_contact = None

    def __post_init__(self):
        self.lamp.init_state.pos = ANYMAL_WORKSPACE.object_pos
