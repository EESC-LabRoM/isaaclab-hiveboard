# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 ball-valve task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.anymal.ball_valve.env import AnymalBallValveEnvCfg, AnymalBallValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka


def _use_franka_ball_valve(env_cfg) -> None:
    """Pinch across the lever's thickness and settle the fingers before turning."""
    use_franka(env_cfg, "ball_valve", "reset_valve_root", finger_close=0.010)
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name in ("approaching", "lever_pivot", "rotate_frame"):
            frame.offset = OffsetCfg(
                pos=frame.offset.pos,
                rot=(0.0, 0.7071068, 0.7071068, 0.0),
            )


@configclass
class FrankaBallValveEnvCfg(AnymalBallValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard ball-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_ball_valve(self)


@configclass
class FrankaBallValveEnvCfg_PLAY(AnymalBallValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_ball_valve(self)
