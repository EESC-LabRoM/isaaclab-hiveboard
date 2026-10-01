# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Binary gripper action whose joint target ramps instead of jumping."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils.configclass import configclass


class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
    """Move the position target toward open/close at a capped speed.

    The stock action steps the target from open to close in one physics step,
    so the PD drive slams the fingers into the part at full effort. Here the
    target advances by at most ``speed * dt`` per call, and the drive only ever
    sees a small tracking error.

    Isaac Lab calls ``apply_actions`` once per physics step, except when the
    physics backend runs the decimation loop itself (Newton): then it is called
    once per env step, and ``dt`` is the env step.
    """

    cfg: RateLimitedBinaryJointPositionActionCfg

    def __init__(self, cfg: RateLimitedBinaryJointPositionActionCfg, env):
        super().__init__(cfg, env)
        per_env_step = getattr(env, "_physics_handles_decimation", False)
        self._dt = float(env.step_dt if per_env_step else env.physics_dt)
        self._target = self._open_command.expand(self.num_envs, -1).clone()
        # Plain indices for reading measured joints (``_joint_ids`` may be a Warp array).
        self._measured_ids = self._asset.find_joints(self._joint_names, preserve_order=True)[0]

    def apply_actions(self):
        goal = self._processed_actions
        closing = (goal - self._open_command).abs() > (self._target - self._open_command).abs()
        speed = torch.where(
            closing,
            _speed(self.cfg.close_speed),
            _speed(self.cfg.open_speed),
        )
        step = speed * self._dt
        # In place: rebinding here under inference_mode would turn the buffer into
        # an inference tensor, and the in-place write in reset() would then fail.
        self._target += torch.clamp(goal - self._target, -step, step)
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # Start from where the fingers are: open after a normal reset, closed
        # when an episode starts mid-grasp. The action manager resets after
        # the reset events, so the measured joints hold the reset state.
        self._target[ids] = self._asset.data.joint_pos.torch[ids][:, self._measured_ids]


def _speed(value: float | None) -> float:
    return float("inf") if value is None else float(value)


@configclass
class RateLimitedBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Binary joint-position action with capped target speeds [rad/s or m/s]."""

    class_type: type = RateLimitedBinaryJointPositionAction

    close_speed: float | None = None
    """Target speed while closing. ``None`` jumps straight to the close command."""
    open_speed: float | None = None
    """Target speed while opening. ``None`` jumps straight to the open command."""
