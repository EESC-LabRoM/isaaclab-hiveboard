# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 peg insertion task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.peg_insertion.env import AnymalPegInsertionEnvCfg, AnymalPegInsertionEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import shift_target_frames, use_franka
from isaaclab_hiveboard.tasks.scenes.peg_insertion import PEG_SPEC
from isaaclab_hiveboard.tasks.scenes.screw import SCREW_GRASP_FRAMES


def _use_franka_peg_insertion(env_cfg) -> None:
    use_franka(env_cfg, "peg", "reset_object_root")
    shift_target_frames(env_cfg, SCREW_GRASP_FRAMES, PEG_SPEC.franka_tip_x - PEG_SPEC.grasp_x)


@configclass
class FrankaPegInsertionEnvCfg(AnymalPegInsertionEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard peg insertion scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_peg_insertion(self)


@configclass
class FrankaPegInsertionEnvCfg_PLAY(AnymalPegInsertionEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_peg_insertion(self)
