# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 hidden-button task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.button.env import AnymalButtonEnvCfg, AnymalButtonEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka

# Object y of the FR3's lid pinch. The shared -0.018 sits on the lid's rounded
# free end, where the FR3's independent fingers squeeze a taper and shove the
# compliant arm sideways; pinch where the lid is full width instead.
LID_PINCH_Y = -0.005


def _use_franka_button(env_cfg) -> None:
    use_franka(env_cfg, "button", "reset_object_root")
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("lid_approaching", "lid_grasp"):
            x, _, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x, LID_PINCH_Y, z), rot=frame.offset.rot)


@configclass
class FrankaButtonEnvCfg(AnymalButtonEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard hidden-button scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_button(self)


@configclass
class FrankaButtonEnvCfg_PLAY(AnymalButtonEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_button(self)
