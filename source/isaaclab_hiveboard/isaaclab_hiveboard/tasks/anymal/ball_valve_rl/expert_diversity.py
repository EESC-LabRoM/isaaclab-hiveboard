# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode variations of the scripted cuRobo expert, for a diverse expert bank.

A reset event (:class:`sample_expert_diversity`) samples, per environment,
the values in :data:`EXPERT_DIVERSITY` and writes them into the
``pose_command`` handlers before they plan:

* ``turn_rate`` - the valve arc's angular speed [rad/s];
* ``reach_scale`` - a factor on the approach and grasp segments' speeds;
* ``grasp_shift`` - where along the lever the TCP grasps [m];
* ``grasp_roll`` - the grasp's rotation about the lever's long axis [rad],
  i.e. the side the hand comes from around the bar;
* ``turn_overshoot`` - not sampled: how far past 90 deg the arc runs [rad].
  Under load the lever lags the hand by ~0.02 rad per N·m (mostly the arm's
  compliance), so the expert turns further the harder the valve
  (:func:`turn_overshoot`) and the valve's end stop holds the lever at open.
  Without it the expert opened 38% of the 3-4 N·m valves; with it 85%.

The grasp offset is applied in the grasp goal's frame, whose ``y`` axis runs
along the lever (the goal sits 6 cm out along it from the valve axis). The
approach goals before the grasp get the same offset expressed from their own
frame, so the final approach keeps its direction relative to the grasp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Order of :attr:`sample_expert_diversity.values` and the bank's ``expert_diversity``.
EXPERT_DIVERSITY = ("turn_rate", "reach_scale", "grasp_shift", "grasp_roll", "turn_overshoot")
#: Sampling ranges. The authored expert turns at 0.3 rad/s with ``reach_scale``
#: 1 and no grasp offset. Slowest case: a ~4 s reach and a 6.3 s turn, open by
#: ~10.5 s of the RL task's 12 s episode.
EXPERT_DIVERSITY_RANGES = {
    "turn_rate": (0.25, 0.5),
    "reach_scale": (0.75, 1.3),
    "grasp_shift": (-0.015, 0.015),
    "grasp_roll": (-0.25, 0.25),
}
#: Arc overshoot [rad] for a free valve, and per N·m of torque at open.
OVERSHOOT_RAD = 0.03
OVERSHOOT_RAD_PER_NM = 0.05
#: Lever axis in the grasp goal's frame.
LEVER_AXIS = (0.0, 1.0, 0.0)


def turn_overshoot(env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Arc overshoot [rad] for ``env_ids`` from the torque their valve needs at open (``valve_dynamics.py``)."""
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

    term = getattr(env, "valve_dynamics_term", None)
    if term is None:
        return torch.full((len(env_ids),), OVERSHOOT_RAD, device=env.device)
    friction, _, spring, _, _ = term.params[env_ids].unbind(dim=-1)
    torque = friction + spring * abs(mdp.VALVE_OPEN_RAD - mdp.VALVE_CLOSED_RAD)
    return OVERSHOOT_RAD + OVERSHOOT_RAD_PER_NM * torque


class sample_expert_diversity(ManagerTermBase):
    """Reset event: sample :data:`EXPERT_DIVERSITY` per environment and apply it to the scripted expert.

    The ``pose_command`` handlers are found on the first call (the command
    manager is built after the event manager): the rotate segment, the
    go-to segments before it (the reach), and the last of those (the grasp).
    ``values`` holds every environment's current sample, ``(num_envs, 5)``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.values = torch.zeros(env.num_envs, len(EXPERT_DIVERSITY), device=env.device)
        self._rotate = None
        env.expert_diversity_term = self

    def _resolve(self, env: ManagerBasedEnv, command_name: str) -> None:
        from isaaclab_hiveboard.mdp.commands.sequential_pose_command import _GoToFrameHandler, _RotateFrameHandler

        handlers = env.command_manager.get_term(command_name)._command_handlers
        rotate = next(i for i, h in enumerate(handlers) if isinstance(h, _RotateFrameHandler))
        self._rotate = handlers[rotate]
        self._reach = [h for h in handlers[:rotate] if isinstance(h, _GoToFrameHandler)]
        self._grasp = self._reach[-1]
        # Each earlier goal in the grasp goal's frame, K, from the frames'
        # authored offsets. Reading the frame transformer here would latch its
        # pre-reset poses for this step, and the command would then plan to
        # where the lever was in the previous episode.
        grasp = self._goal_in_body(self._grasp, env.device)
        self._approach_in_grasp = [
            math_utils.subtract_frame_transforms(*grasp, *self._goal_in_body(h, env.device)) for h in self._reach[:-1]
        ]

    @staticmethod
    def _goal_in_body(handler, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A go-to handler's goal (frame offset, then its target offset) in the frame's body, ``(1, 3)``/``(1, 4)`` xyzw."""
        cfg = handler._frame.cfg
        frame = next(f for f in cfg.target_frames if f.name == handler.cfg.target_frame_name)
        if frame.prim_path != cfg.target_frames[0].prim_path:
            raise ValueError("Expert diversity needs the reach goals to be offsets of one body.")
        return math_utils.combine_frame_transforms(
            torch.tensor([frame.offset.pos], device=device),
            torch.tensor([frame.offset.rot], device=device),
            torch.tensor([handler.cfg.target_offset_pos], device=device),
            torch.tensor([handler.cfg.target_offset_rot], device=device),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        ranges: dict[str, tuple[float, float]],
        command_name: str = "pose_command",
    ) -> None:
        if self._rotate is None:
            self._resolve(env, command_name)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        n = len(ids)
        values = torch.empty(n, len(EXPERT_DIVERSITY), device=env.device)
        for i, name in enumerate(EXPERT_DIVERSITY[:-1]):
            values[:, i].uniform_(*ranges[name])
        # The valve dynamics event runs first.
        values[:, -1] = turn_overshoot(env, ids)
        self.values[ids] = values
        turn_rate, reach_scale, shift, roll, overshoot = values.unbind(dim=-1)

        self._rotate.speed_scale[ids] = turn_rate / self._rotate.cfg.angular_velocity
        self._rotate.angle_extra_rad[ids] = overshoot
        for handler in self._reach:
            handler.speed_scale[ids] = reach_scale

        axis = torch.tensor(LEVER_AXIS, device=env.device).expand(n, 3)
        grasp_pos = axis * shift[:, None]
        grasp_quat = math_utils.quat_from_angle_axis(roll, axis)
        self._grasp.env_offset_pos[ids] = grasp_pos
        self._grasp.env_offset_rot[ids] = grasp_quat
        # Earlier goals move rigidly with the grasp: with K the earlier goal in
        # the grasp goal's frame, its offset is K^-1 * E * K.
        for handler, (k_pos, k_quat) in zip(self._reach[:-1], self._approach_in_grasp):
            k_pos, k_quat = k_pos.expand(n, 3), k_quat.expand(n, 4)
            moved = math_utils.combine_frame_transforms(grasp_pos, grasp_quat, k_pos, k_quat)
            handler.env_offset_pos[ids], handler.env_offset_rot[ids] = math_utils.subtract_frame_transforms(
                k_pos, k_quat, *moved
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def sample_expert_diversity_cfg(ranges: dict[str, tuple[float, float]] | None = None) -> EventTermCfg:
    """Reset event sampling the expert's variations from ``ranges`` (default :data:`EXPERT_DIVERSITY_RANGES`)."""
    return EventTermCfg(
        func=sample_expert_diversity, mode="reset", params={"ranges": dict(ranges or EXPERT_DIVERSITY_RANGES)}
    )
