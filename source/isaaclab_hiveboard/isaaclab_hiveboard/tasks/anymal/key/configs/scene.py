# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, make_ee_frame
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.tasks.anymal.mechanism import pad_contacts
from isaaclab_hiveboard.tasks.scenes.key import KeySceneCfg as KeySceneBase
from isaaclab_hiveboard.tasks.scenes.key import key_frames, key_in_hand

_LEFT_PAD, _RIGHT_PAD = pad_contacts("Lock")
# Glue the key with the 2F-140 rolled a quarter turn, so the pads face the
# bow's flat sides. +90 keeps the turn inside dynaarm_wrist_rotation's
# [-90, 270] deg range: at 0 the turn ran into the -90 deg stop and cuRobo
# jumped to another arm configuration with the key in the lock.
ANYMAL_KEY_JAW_ROLL_DEG = 90.0


@configclass
class KeySceneCfg(KeySceneBase):
    """ANYmal + DynaArm, key glued in the 2F-140, + shared HiveBoard lock scene."""

    robot: ArticulationCfg = key_in_hand(
        ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"), ANYMAL_EE, jaw_roll_deg=ANYMAL_KEY_JAW_ROLL_DEG
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    finger_contact: ContactSensorCfg = _LEFT_PAD
    jaw_contact: ContactSensorCfg = _RIGHT_PAD
    target_frame: FrameTransformerCfg = key_frames(jaw_roll_deg=ANYMAL_KEY_JAW_ROLL_DEG)
