# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import SPOT_EE, make_ee_frame
from isaaclab_hiveboard.assets.spot.spot import SPOT_ARM_CFG
from isaaclab_hiveboard.tasks.scenes.button import ButtonSceneCfg as ButtonSceneBase


@configclass
class ButtonSceneCfg(ButtonSceneBase):
    """Spot + shared HiveBoard hidden-button scene."""

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_EE)
