# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset from, and track, precomputed cuRobo expert trajectories.

The bank (``scripts/rl/build_expert_bank.py``) holds successful scripted-expert
rollouts from the RL task's own reset distribution. Each episode is reset to the
initial state of one bank trajectory, which then provides the arm-joint
reference for that episode:

* **Reach** (before the lever is first held): the expert's *planned* joint
  path (cuRobo's joint targets) indexed by episode time, up to the grasp. Steps
  where the plan does not move - the scripted expert settling onto each
  segment's end pose before starting the next - are dropped, so the reference
  moves at the plan's speeds with no pauses.
* **Turn** (once held): the expert's joints as a function of valve angle,
  evaluated at the valve's actual angle. It asks the policy to hold and turn
  the lever with the expert's joint configuration; *when* to turn is scored
  separately against the turning-rate command's reference angle. (Evaluating
  it at the reference angle instead put a constant ~0.1 rad lead into the
  joint reference, because the expert pauses ~0.5 s for its gripper to close.)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import BaseArticulation
from isaaclab.managers import EventTermCfg, ManagerTermBase

from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

#: Samples of the turn reference over valve angle.
TURN_GRID = 64
#: Largest joint-target change [rad] of a step counted as idle in the reach.
IDLE_STEP_RAD = 1.0e-3


class ExpertBank:
    """Expert references precomputed per trajectory, on the simulation device."""

    def __init__(self, path: str, device: str, valve_open_rad: float):
        bank = torch.load(path, map_location="cpu", weights_only=False)
        meta = bank["meta"]
        if list(meta["arm_joint_names"]) != list(ANYMAL_ARM_JOINT_NAMES):
            raise ValueError(f"Bank joint order {meta['arm_joint_names']} differs from {ANYMAL_ARM_JOINT_NAMES}")
        self.meta = meta
        self.size = len(bank["arm_q"])
        q = bank["arm_q"].float()  # (N, T, 6)
        valve = bank["valve_angle"].float()  # (N, T)
        closed = bank["gripper_cmd"] < 0
        # Grasp step: first closed-gripper step. Open step: first step at open.
        grasp = closed.float().argmax(dim=1)
        open_step = (bank["t_open"] / meta["dt"]).round().long().clamp(max=q.shape[1] - 1)

        # Reach: planned joints up to the grasp without the idle steps, then
        # held at the grasp configuration.
        target = bank["arm_target"].float()
        paths = []
        for i in range(self.size):
            path = target[i, : int(grasp[i]) + 1]
            moving = (path[1:] - path[:-1]).abs().amax(dim=-1) > IDLE_STEP_RAD
            paths.append(torch.cat((path[:1], path[1:][moving])))
        horizon = max(len(path) for path in paths)
        self.reach = torch.stack([torch.cat((p, p[-1:].expand(horizon - len(p), -1))) for p in paths])  # (N, H, 6)
        self.grasp_step = torch.tensor([len(p) - 1 for p in paths])

        # Turn: joints sampled on a uniform valve-angle grid from the grasp
        # angle to open, by linear interpolation over the expert's samples.
        theta0 = valve[torch.arange(self.size), grasp]
        grid = theta0[:, None] + (valve_open_rad - theta0)[:, None] * torch.linspace(0, 1, TURN_GRID)[None]
        self.turn = torch.empty(self.size, TURN_GRID, q.shape[-1])
        for i in range(self.size):
            seg = slice(int(grasp[i]), int(open_step[i]) + 1)
            # The valve angle falls monotonically while turning; flip to ascending.
            angles = -valve[i, seg]
            angles = torch.cummax(angles, dim=0).values
            joints = q[i, seg]
            self.turn[i] = _interp(-grid[i], angles, joints)
        self.turn_theta0 = theta0
        self.valve_open_rad = valve_open_rad

        self.arm_q0 = bank["arm_q0"].float()
        self.valve_angle0 = bank["valve_angle0"].float()
        self.valve_pose_env = bank["valve_pose_env"].float()
        for name in ("reach", "grasp_step", "turn", "turn_theta0", "arm_q0", "valve_angle0", "valve_pose_env"):
            setattr(self, name, getattr(self, name).to(device))

    def reach_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert joints at episode ``step`` (held at the grasp afterwards), ``(len(idx), 6)``."""
        return self.reach[idx, step.clamp(max=self.reach.shape[1] - 1)]

    def turn_reference(self, idx: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Expert joints at valve ``angle`` [rad] along the turn, ``(len(idx), 6)``."""
        theta0 = self.turn_theta0[idx]
        frac = ((angle - theta0) / (self.valve_open_rad - theta0)).clamp(0.0, 1.0) * (TURN_GRID - 1)
        lo = frac.floor().long().clamp(max=TURN_GRID - 2)
        w = (frac - lo.float())[:, None]
        return self.turn[idx, lo] * (1.0 - w) + self.turn[idx, lo + 1] * w


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear interpolation of rows ``fp`` (len(xp), D) at ``x``; clamped at the ends."""
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    w = ((x - x0) / (x1 - x0).clamp(min=1.0e-9)).clamp(0.0, 1.0)[:, None]
    return fp[idx - 1] * (1.0 - w) + fp[idx] * w


class reset_from_expert_bank(ManagerTermBase):
    """Reset event: start each episode at a random bank trajectory's initial state.

    Writes the valve root pose, the valve angle and the arm joints of the drawn
    trajectory (the scene's other joints keep the preceding default reset) and
    remembers the index, which the reference terms read. Replaces the valve
    pose, valve angle and arm reset randomization, which the bank already
    samples from the same distribution.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

        self.bank = ExpertBank(cfg.params["path"], env.device, mdp.VALVE_OPEN_RAD)
        self.index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robot: BaseArticulation = env.scene["robot"]
        self._arm_ids = robot.find_joints(list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)[0]
        valve: BaseArticulation = env.scene["ball_valve"]
        self._valve_joint = valve.find_joints(mdp.VALVE_JOINT)[0]
        # Reference terms find the bank here.
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, path: str) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        draw = torch.randint(0, self.bank.size, (len(ids),), device=env.device)
        self.write_start_state(env, ids, draw)

    def write_start_state(self, env: ManagerBasedEnv, ids: torch.Tensor, draw: torch.Tensor) -> None:
        """Put environments ``ids`` at the initial state of bank trajectories ``draw``."""
        self.index[ids] = draw
        robot: BaseArticulation = env.scene["robot"]
        valve: BaseArticulation = env.scene["ball_valve"]
        pose = self.bank.valve_pose_env[draw].clone()
        pose[:, :3] += env.scene.env_origins[ids]
        valve.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        valve.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        valve.write_joint_position_to_sim_index(
            position=self.bank.valve_angle0[draw, None], joint_ids=self._valve_joint, env_ids=ids
        )
        valve.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), 1, device=env.device), joint_ids=self._valve_joint, env_ids=ids
        )
        robot.write_joint_position_to_sim_index(position=self.bank.arm_q0[draw], joint_ids=self._arm_ids, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._arm_ids), device=env.device), joint_ids=self._arm_ids, env_ids=ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def _bank_term(env: ManagerBasedEnv) -> reset_from_expert_bank:
    term = getattr(env, "expert_bank_term", None)
    if term is None:
        raise RuntimeError("Expert-bank terms need the reset_from_expert_bank event.")
    return term


def expert_joint_reference(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Expert arm-joint reference for the current phase, ``(N, 6)`` [rad].

    Reach reference by episode time until the turning command engages at the
    first grasp; afterwards the turn reference at the valve's actual angle.
    """
    term = _bank_term(env)
    idx = term.index
    command = env.command_manager.get_term(command_name)
    reach = term.bank.reach_reference(idx, env.episode_length_buf)
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

    turn = term.bank.turn_reference(idx, mdp.valve_angle(env))
    return torch.where(command.engaged[:, None], turn, reach)


def expert_joint_error(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Arm joints minus the expert reference, ``(N, 6)`` [rad] (privileged observation)."""
    robot: BaseArticulation = env.scene["robot"]
    arm = _bank_term(env)._arm_ids
    return robot.data.joint_pos.torch[:, arm] - expert_joint_reference(env, command_name)


def track_expert_joints(env: ManagerBasedRLEnv, std: float, command_name: str = "valve_turn") -> torch.Tensor:
    """``exp(-||q - q_expert||^2 / std^2)`` over the six arm joints; ``std`` [rad]."""
    error = expert_joint_error(env, command_name)
    return torch.exp(-error.square().sum(dim=-1) / std**2)


def reset_from_expert_bank_cfg(path: str) -> EventTermCfg:
    """Reset event drawing each episode's start from the bank at ``path``."""
    return EventTermCfg(func=reset_from_expert_bank, mode="reset", params={"path": path})


def expert_gripper_reference(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """1 where the expert's gripper is closed, ``(N, 1)`` (privileged observation).

    Closed from the expert's grasp step on the idle-free reach timeline (see
    :class:`ExpertBank`), and once the turning command has engaged.
    """
    term = _bank_term(env)
    command = env.command_manager.get_term(command_name)
    closed = (env.episode_length_buf >= term.bank.grasp_step[term.index]) | command.engaged
    return closed.float().unsqueeze(-1)


def track_expert_gripper(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """1 while the policy's gripper command (open/close) matches the expert's."""
    policy_closed = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0
    expert_closed = expert_gripper_reference(env, command_name)[:, 0].bool()
    return (policy_closed == expert_closed).float()
