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
* **Turn** (from the expert's grasp on): the expert's measured joints, also
  by episode time, held at the pose where it opened the valve (the scripted
  expert lets go afterwards). An earlier version evaluated the expert's joints
  at the valve's *actual* angle, so a lever held still kept the arm exactly on
  the reference and joint tracking paid for not turning (teachers v17/v18).

Before the grasp, episode time is the reach's idle-free timeline; the turn
and the valve reference are shifted by the idle steps dropped from the reach.
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

#: Every joint of the Robotiq linkage. The bank stores all of them so an
#: episode can start mid-grasp without the solver snapping the linkage.
GRIPPER_LINKAGE_JOINTS = [
    "finger_joint",
    "left_outer_finger_joint",
    "left_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_outer_knuckle_joint",
    "right_outer_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_pad_joint",
]
#: Pad force [N] above which a finger pad counts as touching the lever. The
#: expert's pads read 0 N before the grasp and 8-14 N while turning.
CONTACT_FORCE_N = 1.0
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

        # Turn and valve: the expert's measured joints and valve angle by time.
        # Episode steps past the reach's grasp map to the bank's own steps by
        # the idle steps the reach dropped.
        self.q = q
        self.valve = valve
        self.gripper_q = bank["gripper_q"].float()
        if "pad_force" not in bank:
            raise ValueError(f"{path} has no pad_force; rebuild it with scripts/rl/build_expert_bank.py")
        # Per-pad contact with the lever, (N, T, 2).
        self.contact = (bank["pad_force"].float() > CONTACT_FORCE_N).float()
        if list(meta.get("gripper_joint_names", [])) != GRIPPER_LINKAGE_JOINTS:
            raise ValueError(f"{path} has no gripper_joint_pos; rebuild it with scripts/rl/build_expert_bank.py")
        # The whole Robotiq linkage, for starting episodes mid-grasp, (N, T, 8).
        self.gripper_joint_pos = bank["gripper_joint_pos"].float()
        self.dt = float(meta["dt"])
        self.idle_shift = grasp - self.grasp_step
        self.open_step = open_step
        self.valve_open_rad = valve_open_rad
        self.arm_q0 = bank["arm_q0"].float()
        self.valve_angle0 = bank["valve_angle0"].float()
        self.valve_pose_env = bank["valve_pose_env"].float()
        for name in (
            "reach",
            "grasp_step",
            "q",
            "valve",
            "gripper_q",
            "contact",
            "gripper_joint_pos",
            "idle_shift",
            "open_step",
            "arm_q0",
            "valve_angle0",
            "valve_pose_env",
        ):
            setattr(self, name, getattr(self, name).to(device))

    def reach_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert joints at episode ``step`` (held at the grasp afterwards), ``(len(idx), 6)``."""
        return self.reach[idx, step.clamp(max=self.reach.shape[1] - 1)]

    def _bank_step(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """The bank's own step for episode ``step``, held at the expert's open step."""
        shifted = torch.where(step >= self.grasp_step[idx], step + self.idle_shift[idx], step)
        return torch.minimum(shifted, self.open_step[idx])

    def valve_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert valve angle at episode ``step`` (held once open), ``(len(idx),)`` [rad]."""
        return self.valve[idx, self._bank_step(idx, step)]

    def gripper_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert finger joint position at episode ``step`` (held once open), ``(len(idx),)`` [rad]."""
        return self.gripper_q[idx, self._bank_step(idx, step)]

    def contact_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert per-pad lever contact (1/0) at episode ``step`` (held once open), ``(len(idx), 2)``."""
        return self.contact[idx, self._bank_step(idx, step)]

    def turn_reference(self, idx: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """Expert measured joints at episode ``step`` (held once open), ``(len(idx), 6)``."""
        return self.q[idx, self._bank_step(idx, step)]


class reset_from_expert_bank(ManagerTermBase):
    """Reset event: start each episode on a random bank trajectory.

    Writes the valve root pose, the valve angle and the arm joints of the drawn
    trajectory (the scene's other joints keep the preceding default reset) and
    remembers the index, which the reference terms read. Replaces the valve
    pose, valve angle and arm reset randomization, which the bank already
    samples from the same distribution.

    With probability ``mid_start_prob`` the episode starts at a random step of
    the trajectory up to the expert's open step instead of its beginning
    (reference state initialization): arm, whole gripper linkage and valve as
    the expert had them, e.g. already gripping and turning. ``start_step``
    holds that step; :func:`reference_step` adds it to the episode time, so
    every time-indexed reference starts there too. Students trained from the
    beginning only (v9-v11, PPO v1) never discovered closing the gripper.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

        self.bank = ExpertBank(cfg.params["path"], env.device, mdp.VALVE_OPEN_RAD)
        self.index = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.start_step = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        robot: BaseArticulation = env.scene["robot"]
        self._hand_ids = robot.find_joints(GRIPPER_LINKAGE_JOINTS, preserve_order=True)[0]
        self._arm_ids = robot.find_joints(list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)[0]
        # The joint the bank recorded as gripper_q (build_expert_bank.py).
        self._finger_id = robot.find_joints(env.cfg.actions.gripper_action.joint_names)[0][0]
        valve: BaseArticulation = env.scene["ball_valve"]
        self._valve_joint = valve.find_joints(mdp.VALVE_JOINT)[0]
        # Reference terms find the bank here.
        env.expert_bank_term = self
        print(f"[INFO] Expert bank: {self.bank.size} trajectories from {cfg.params['path']}")

    def __call__(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor | None, path: str, mid_start_prob: float = 0.0
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        draw = torch.randint(0, self.bank.size, (len(ids),), device=env.device)
        self.write_start_state(env, ids, draw)
        if mid_start_prob > 0.0:
            mid = torch.rand(len(ids), device=env.device) < mid_start_prob
            if torch.any(mid):
                last = self.bank.open_step[draw[mid]] - self.bank.idle_shift[draw[mid]]
                step = (torch.rand(int(mid.sum()), device=env.device) * (last + 1).float()).long()
                self.write_mid_state(env, ids[mid], draw[mid], step)

    def write_start_state(self, env: ManagerBasedEnv, ids: torch.Tensor, draw: torch.Tensor) -> None:
        """Put environments ``ids`` at the initial state of bank trajectories ``draw``."""
        self.index[ids] = draw
        self.start_step[ids] = 0
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

    def write_mid_state(self, env: ManagerBasedEnv, ids: torch.Tensor, draw: torch.Tensor, step: torch.Tensor) -> None:
        """Put environments ``ids`` at episode ``step`` of bank trajectories ``draw`` (after the start state)."""
        bank = self.bank
        self.start_step[ids] = step
        b = bank._bank_step(draw, step)
        prev, nxt = (b - 1).clamp(min=0), (b + 1).clamp(max=bank.q.shape[1] - 1)
        span = ((nxt - prev).clamp(min=1) * bank.dt)[:, None]
        robot: BaseArticulation = env.scene["robot"]
        valve: BaseArticulation = env.scene["ball_valve"]
        robot.write_joint_position_to_sim_index(position=bank.q[draw, b], joint_ids=self._arm_ids, env_ids=ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=(bank.q[draw, nxt] - bank.q[draw, prev]) / span, joint_ids=self._arm_ids, env_ids=ids
        )
        robot.write_joint_position_to_sim_index(
            position=bank.gripper_joint_pos[draw, b], joint_ids=self._hand_ids, env_ids=ids
        )
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros(len(ids), len(self._hand_ids), device=env.device),
            joint_ids=self._hand_ids,
            env_ids=ids,
        )
        valve.write_joint_position_to_sim_index(
            position=bank.valve[draw, b, None], joint_ids=self._valve_joint, env_ids=ids
        )
        valve.write_joint_velocity_to_sim_index(
            velocity=(bank.valve[draw, nxt] - bank.valve[draw, prev])[:, None] / span,
            joint_ids=self._valve_joint,
            env_ids=ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def reference_step(env: ManagerBasedEnv) -> torch.Tensor:
    """Step along the expert trajectory: the episode step plus where the episode started on it."""
    term = _bank_term(env)
    return env.episode_length_buf + term.start_step


def _bank_term(env: ManagerBasedEnv) -> reset_from_expert_bank:
    term = getattr(env, "expert_bank_term", None)
    if term is None:
        raise RuntimeError("Expert-bank terms need the reset_from_expert_bank event.")
    return term


def expert_joint_reference(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Expert arm-joint reference at the current episode time, ``(N, 6)`` [rad].

    The reach until the expert's grasp step, then the expert's turn (see the
    module docstring). ``command_name`` is kept for the callers' signature.
    """
    term = _bank_term(env)
    idx, step = term.index, reference_step(env)
    reach = term.bank.reach_reference(idx, step)
    turn = term.bank.turn_reference(idx, step)
    return torch.where((step >= term.bank.grasp_step[idx])[:, None], turn, reach)


def expert_joint_error(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Arm joints minus the expert reference, ``(N, 6)`` [rad] (privileged observation)."""
    robot: BaseArticulation = env.scene["robot"]
    arm = _bank_term(env)._arm_ids
    return robot.data.joint_pos.torch[:, arm] - expert_joint_reference(env, command_name)


def track_expert_joints(env: ManagerBasedRLEnv, std: float, command_name: str = "valve_turn") -> torch.Tensor:
    """``exp(-||q - q_expert||^2 / std^2)`` over the six arm joints; ``std`` [rad]."""
    error = expert_joint_error(env, command_name)
    return torch.exp(-error.square().sum(dim=-1) / std**2)


def expert_valve_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Valve angle minus the expert's at the same episode time, ``(N, 1)`` [rad] (privileged observation)."""
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

    term = _bank_term(env)
    return (mdp.valve_angle(env) - term.bank.valve_reference(term.index, reference_step(env))).unsqueeze(-1)


def track_expert_valve(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``exp(-(θ - θ_expert(t))^2 / std^2)``: the valve follows the expert's by episode time; ``std`` [rad].

    Before the expert grasps, this pays for leaving the valve where it is;
    after, for turning it on the expert's schedule; once the expert has
    opened, for keeping it open.
    """
    return torch.exp(-expert_valve_error(env)[:, 0].square() / std**2)


def reset_from_expert_bank_cfg(path: str, mid_start_prob: float = 0.0) -> EventTermCfg:
    """Reset event drawing each episode's start from the bank at ``path`` (see :class:`reset_from_expert_bank`)."""
    return EventTermCfg(
        func=reset_from_expert_bank, mode="reset", params={"path": path, "mid_start_prob": mid_start_prob}
    )


def expert_gripper_reference(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """1 where the expert's gripper is closed, ``(N, 1)`` (privileged observation).

    Closed from the expert's grasp step on the reach timeline (see
    :class:`ExpertBank`). ``command_name`` is kept for the callers' signature.
    """
    term = _bank_term(env)
    closed = reference_step(env) >= term.bank.grasp_step[term.index]
    return closed.float().unsqueeze(-1)


def track_expert_gripper(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``exp(-(q_finger - q_expert(t))^2 / std^2)``: the fingers follow the expert's by episode time; ``std`` [rad].

    Tracks where the fingers are, not the open/close command: scoring the
    command's sign let teacher v21 cage the lever with half-closed fingers
    (0.2-0.54 rad against the expert's 0.68 on the lever) and reopen one step
    in seven while still earning 6/7 of the reward.
    """
    term = _bank_term(env)
    robot: BaseArticulation = env.scene["robot"]
    finger = robot.data.joint_pos.torch[:, term._finger_id]
    error = finger - term.bank.gripper_reference(term.index, reference_step(env))
    return torch.exp(-error.square() / std**2)


def track_expert_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the two finger pads whose lever contact matches the expert's at this episode time.

    Pays for gripping the lever when the expert does (both pads over
    ``CONTACT_FORCE_N``) and for not touching it before. Nothing else pays
    for contact itself, while contact can disturb the tracked joints and valve.
    """
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp

    term = _bank_term(env)
    policy = (mdp.pad_valve_force(env) > CONTACT_FORCE_N).float()
    expert = term.bank.contact_reference(term.index, reference_step(env))
    return (policy == expert).float().mean(dim=-1)


def expert_joint_drift(env: ManagerBasedRLEnv, max_error: float, command_name: str = "valve_turn") -> torch.Tensor:
    """Termination: the arm is more than ``max_error`` [rad] (joint-space norm) from the expert reference."""
    return torch.linalg.vector_norm(expert_joint_error(env, command_name), dim=-1) > max_error


def expert_valve_lag(env: ManagerBasedRLEnv, max_error: float) -> torch.Tensor:
    """Termination: the valve is more than ``max_error`` [rad] off the expert's valve angle at this episode time.

    Behind catches holding the lever without turning it (teacher v17), ahead
    catches whipping it round (teacher v16); before the grasp it catches
    knocking the lever.
    """
    return expert_valve_error(env)[:, 0].abs() > max_error
