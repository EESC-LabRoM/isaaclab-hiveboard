# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 sliding-drawer task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.drawer.env import AnymalDrawerEnvCfg, AnymalDrawerEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka
from isaaclab_hiveboard.tasks.scenes.drawer import DRAWER_GRASP_X

# The FR3 TCP is at the fingertips, not mid-pad: put the tips just off the
# drawer face (x = 0.038) so the pads cover the handle tab.
FRANKA_GRASP_SHIFT = 0.040 - DRAWER_GRASP_X


def _use_franka_drawer(env_cfg) -> None:
    use_franka(env_cfg, "drawer", "reset_object_root")
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("drawer_grasp", "drawer_pulled"):
            x, y, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x + FRANKA_GRASP_SHIFT, y, z), rot=frame.offset.rot)


@configclass
class FrankaDrawerEnvCfg(AnymalDrawerEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard sliding-drawer scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_drawer(self)


@configclass
class FrankaDrawerEnvCfg_PLAY(AnymalDrawerEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_drawer(self)
