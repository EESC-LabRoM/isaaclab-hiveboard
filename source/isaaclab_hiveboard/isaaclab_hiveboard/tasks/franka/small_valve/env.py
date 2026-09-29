# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 small-valve task, retargeted from the ANYmal task of the same scene."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.tasks.anymal.small_valve.env import AnymalSmallValveEnvCfg, AnymalSmallValveEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import use_franka

# The pads meet the ~34.5 mm-radius handwheel rim at fr3_finger_joint ~0.034 m.
# Closing to 0.0 squeezed 34 mm past it: the first pad to reach a rim knob shoved
# the compliant arm off the wheel while the other pad closed on nothing. Stop
# ~4 mm past contact and ramp the close so the pads don't strike at full effort.
FINGER_CLOSE = 0.030
FINGER_CLOSE_SPEED = 0.05


def _use_franka_small_valve(env_cfg) -> None:
    use_franka(env_cfg, "small_valve", "reset_valve_root")
    # The ANYmal task's 8 s ends mid-turn at the FR3's slower valve arc.
    env_cfg.episode_length_s = 20.0
    hand = env_cfg.actions.gripper_action
    env_cfg.actions.gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name=hand.asset_name,
        joint_names=hand.joint_names,
        open_command_expr=hand.open_command_expr,
        close_command_expr={"fr3_finger_joint.*": FINGER_CLOSE},
        close_speed=FINGER_CLOSE_SPEED,
    )


@configclass
class FrankaSmallValveEnvCfg(AnymalSmallValveEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard small-valve scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_small_valve(self)


@configclass
class FrankaSmallValveEnvCfg_PLAY(AnymalSmallValveEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_small_valve(self)
