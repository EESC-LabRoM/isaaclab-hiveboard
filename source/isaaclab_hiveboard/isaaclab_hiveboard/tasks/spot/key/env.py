# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spot lock-and-key task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import SPOT_EE
from isaaclab_hiveboard.tasks.anymal.key.env import AnymalKeyEnvCfg, AnymalKeyEnvCfg_PLAY
from isaaclab_hiveboard.tasks.scenes.key import key_frames, key_in_hand
from isaaclab_hiveboard.tasks.spot.common import use_spot

# Spot's finger swings open along TCP +Z, so with no roll its jaws face the
# bow's flat sides. The quarter turn runs arm_wr1 from about 0 to -1.7 rad,
# inside its +-2.88 rad limits. At -90 it needed -3.2 and cuRobo jumped to
# another arm configuration with the key in the lock.
SPOT_KEY_JAW_ROLL_DEG = 0.0


def _use_spot_key(env_cfg) -> None:
    use_spot(env_cfg)
    env_cfg.scene.robot = key_in_hand(env_cfg.scene.robot, SPOT_EE, jaw_roll_deg=SPOT_KEY_JAW_ROLL_DEG)
    env_cfg.scene.target_frame = key_frames(jaw_roll_deg=SPOT_KEY_JAW_ROLL_DEG)


@configclass
class SpotKeyEnvCfg(AnymalKeyEnvCfg):
    """Fixed-base Spot on the shared HiveBoard lock-and-key scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_spot_key(self)


@configclass
class SpotKeyEnvCfg_PLAY(AnymalKeyEnvCfg_PLAY):
    """Deterministic one-environment Spot demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_spot_key(self)
