# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 lock-and-key task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.key.env import AnymalKeyEnvCfg, AnymalKeyEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka
from isaaclab_hiveboard.tasks.scenes.key import KEY_GRASP_X

# The FR3 TCP is at the fingertips, not mid-pad: put the tips at the bow's
# inner edge (x = 0.049) so the pads cover the bow.
FRANKA_GRASP_SHIFT = 0.052 - KEY_GRASP_X


def _use_franka_key(env_cfg) -> None:
    use_franka(env_cfg, "key", "reset_object_root")
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("key_grasp", "rotate_frame"):
            x, y, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x + FRANKA_GRASP_SHIFT, y, z), rot=frame.offset.rot)


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
