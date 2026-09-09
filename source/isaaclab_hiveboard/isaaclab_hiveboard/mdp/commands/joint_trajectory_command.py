# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of a fixed joint-space trajectory as a command term."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass


class JointTrajectoryCommand(CommandTerm):
    """Emit successive samples from a vendored ``(q_arm, grip)`` clip."""

    cfg: "JointTrajectoryCommandCfg"

    def __init__(self, cfg: "JointTrajectoryCommandCfg", env):
        super().__init__(cfg, env)
        samples = _load_trajectory(cfg.trajectory_path)
        if samples.ndim != 2 or samples.shape[1] != cfg.command_dim:
            raise ValueError(
                f"Trajectory {cfg.trajectory_path} has shape {tuple(samples.shape)}; "
                f"expected (N, {cfg.command_dim})."
            )
        self._traj = samples.to(device=self.device, dtype=torch.float32)
        self._num_samples = int(self._traj.shape[0])
        self._index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._command = torch.zeros(
            self.num_envs, cfg.command_dim, device=self.device, dtype=torch.float32
        )
        self.valve_joint_des = torch.full(
            (self.num_envs,),
            float(cfg.valve_joint_open),
            device=self.device,
            dtype=torch.float32,
        )
        self.valve_task_goal = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._apply_index()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def is_done(self) -> torch.Tensor:
        return self._index >= (self._num_samples - 1)

    def _resample_command(self, env_ids: Sequence[int] | slice | torch.Tensor):
        env_ids = self._resolve_env_ids(env_ids)
        self._index[env_ids] = 0
        self.valve_joint_des[env_ids] = float(self.cfg.valve_joint_open)
        self.valve_task_goal[env_ids] = 1.0
        self._apply_index(env_ids)

    def _update_command(self):
        # In-place so torch.inference_mode() during env.step does not replace
        # ``_index`` with an inference tensor that later blocks reset().
        self._index.add_(1)
        self._index.clamp_(max=self._num_samples - 1)
        self._apply_index()

    def _update_metrics(self):
        return

    def _apply_index(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._command[env_ids] = self._traj[self._index[env_ids]]

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, slice) or env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)


def _load_trajectory(path: str | Path) -> torch.Tensor:
    payload = json.loads(Path(path).read_text())
    samples = payload.get("q", payload.get("qpos"))
    if samples is None:
        raise ValueError(f"Trajectory file {path} has no 'q' or 'qpos' array.")
    return torch.tensor(samples, dtype=torch.float32)


@configclass
class JointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for :class:`JointTrajectoryCommand`."""

    class_type: type = JointTrajectoryCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    trajectory_path: str = ""
    command_dim: int = 7
    valve_asset_name: str | None = "ball_valve"
    valve_joint_name: str = "RevoluteJoint"
    valve_joint_closed: float = 0.0
    valve_joint_open: float = -1.5707963267948966
