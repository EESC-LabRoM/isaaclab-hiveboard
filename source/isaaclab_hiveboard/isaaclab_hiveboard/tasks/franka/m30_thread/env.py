# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 M30 thread task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.m30_thread.env import AnymalM30ThreadEnvCfg, AnymalM30ThreadEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import shift_target_frames, use_franka
from isaaclab_hiveboard.tasks.scenes.screw import SCREW_GRASP_FRAMES
from isaaclab_hiveboard.tasks.scenes.threads import M30_SPEC


def _use_franka_m30_thread(env_cfg) -> None:
    use_franka(env_cfg, "thread", "reset_object_root")
    shift_target_frames(env_cfg, SCREW_GRASP_FRAMES, M30_SPEC.franka_tip_x - M30_SPEC.grasp_x)


@configclass
class FrankaM30ThreadEnvCfg(AnymalM30ThreadEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard M30 thread scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_m30_thread(self)


@configclass
class FrankaM30ThreadEnvCfg_PLAY(AnymalM30ThreadEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_m30_thread(self)
