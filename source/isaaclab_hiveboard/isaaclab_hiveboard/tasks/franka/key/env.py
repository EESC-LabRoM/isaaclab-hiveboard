# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 lock-and-key task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE
from isaaclab_hiveboard.tasks.anymal.key.env import AnymalKeyEnvCfg, AnymalKeyEnvCfg_PLAY, key_contact_stiffness
from isaaclab_hiveboard.tasks.franka.common import use_franka
from isaaclab_hiveboard.tasks.scenes.key import key_frames, key_in_hand

# The FR3 TCP is at the fingertips, not mid-pad: put the tips at the bow's
# inner edge so the pads cover the bow.
FRANKA_KEY_GRASP_X = 0.052
# Per-finger opening that leaves the jaws on the 7 mm thick bow [m]. The key
# is welded to the hand, so this is only for show.
FRANKA_KEY_FINGER_CLOSE = 0.0035
# Roll the key a quarter turn so the pads face the bow's flat sides.
FRANKA_KEY_JAW_ROLL_DEG = 90.0


def _use_franka_key(env_cfg) -> None:
    use_franka(env_cfg, "lock", "reset_object_root", finger_close=FRANKA_KEY_FINGER_CLOSE)
    env_cfg.scene.robot = key_in_hand(env_cfg.scene.robot, FRANKA_EE, FRANKA_KEY_GRASP_X, FRANKA_KEY_JAW_ROLL_DEG)
    env_cfg.scene.target_frame = key_frames(FRANKA_KEY_GRASP_X, FRANKA_KEY_JAW_ROLL_DEG)
    # use_franka replaces every contact-stiffness event with its finger one.
    env_cfg.events.key_contacts = key_contact_stiffness()


@configclass
class FrankaKeyEnvCfg(AnymalKeyEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard lock-and-key scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_key(self)


@configclass
class FrankaKeyEnvCfg_PLAY(AnymalKeyEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_key(self)
