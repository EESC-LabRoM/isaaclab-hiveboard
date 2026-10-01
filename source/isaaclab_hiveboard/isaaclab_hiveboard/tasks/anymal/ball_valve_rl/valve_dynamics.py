# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-episode ball-valve dynamics: how hard the lever is to turn.

Each reset samples five parameters per environment (:data:`VALVE_DYNAMICS`):

* ``friction`` - Coulomb friction [N·m], MJWarp's ``dof_frictionloss``;
* ``damping`` - viscous friction [N·m·s/rad], MJWarp's ``dof_damping``;
* ``spring`` - a return spring toward closed [N·m/rad];
* ``breakaway`` - an extra seat torque [N·m] that holds the lever toward
  closed, full at the seat and fading to zero over
  :data:`BREAKAWAY_WIDTH_RAD` of opening (a ball valve's seat is tightest
  where it starts to open); applied by :class:`ValveLoadAction`;
* ``armature`` - added rotor inertia [kg·m²].

With probability ``stuck_prob`` an episode's valve is instead stuck closed:
the lever starts seated at closed and ``breakaway`` comes from
:data:`STUCK_BREAKAWAY_RANGE`, so it takes several N·m to unseat before it
turns like the others.

Friction, damping and armature are simulator joint properties written on
reset. The spring and seat torques depend on the angle and are applied by
:class:`ValveLoadAction` every env step (writing the valve drive's
stiffness has no effect on Newton). Newton runs the decimation loop itself, so
the torque is held for the step's physics substeps; the lever moves ~0.02 rad
in that time.

Expert trajectories are recorded under the sampled values and store them
(``valve_dynamics`` in the bank); RL episodes restore them with the
trajectory so the time-indexed references stay feasible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import BaseArticulation
from isaaclab.managers import ActionTerm, ActionTermCfg, EventTermCfg, ManagerTermBase
from isaaclab.utils.configclass import configclass

from .mdp import VALVE_CLOSED_RAD, VALVE_JOINT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Parameter order in :attr:`randomize_valve_dynamics.params` and the bank.
VALVE_DYNAMICS = ("friction", "damping", "spring", "breakaway", "armature")
#: Sampling ranges. The defaults (``friction`` 0.02 N·m, the rest 0) turn with
#: almost no effort; these reach a stiff valve (several N·m at open) that the
#: ~100 N force-limited grip (``env.py``) still holds.
VALVE_DYNAMICS_RANGES = {
    "friction": (0.05, 2.0),
    "damping": (0.0, 0.5),
    "spring": (0.0, 1.0),
    "breakaway": (0.0, 2.0),
    "armature": (0.001, 0.02),
}
#: Seat torque [N·m] of a valve stuck closed, and how often that happens.
STUCK_BREAKAWAY_RANGE = (2.0, 5.0)
STUCK_PROB = 0.25
#: :func:`valve_dynamics_obs` scale: each parameter's largest sampled value.
VALVE_DYNAMICS_SCALE = tuple(
    max(VALVE_DYNAMICS_RANGES[name][1], STUCK_BREAKAWAY_RANGE[1] if name == "breakaway" else 0.0)
    for name in VALVE_DYNAMICS
)
#: Opening [rad] over which the seat torque fades from ``breakaway`` to 0.
BREAKAWAY_WIDTH_RAD = 0.15
#: Opening [rad] up to which the spring may not overcome friction. A lever
#: reset part-open (down to -0.4 rad) then only creeps until it is grasped
#: (MJWarp's friction is a soft constraint: 0.15 N·m held below friction
#: creeps at ~0.03 rad/s whatever the friction), instead of snapping shut
#: from under the expert's planned grasp.
SPRING_STATIC_RAD = 0.4


def closing_torque(angle: torch.Tensor, spring: torch.Tensor, breakaway: torch.Tensor) -> torch.Tensor:
    """Spring plus seat torque [N·m] at valve ``angle`` [rad]; positive turns the lever toward closed."""
    opening = (VALVE_CLOSED_RAD - angle).clamp(min=0.0)
    return spring * opening + breakaway * (1.0 - opening / BREAKAWAY_WIDTH_RAD).clamp(min=0.0)


class randomize_valve_dynamics(ManagerTermBase):
    """Reset event: sample :data:`VALVE_DYNAMICS` per environment and write them to the valve.

    ``params`` holds every environment's current values, ``(num_envs, 5)``;
    :meth:`write` sets given values instead (the expert-bank reset uses it).
    The spring is capped so that it cannot overcome friction up to
    :data:`SPRING_STATIC_RAD` of opening. A stuck valve's lever is moved to
    closed, so this event runs after the valve joint's reset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.valve: BaseArticulation = env.scene[cfg.params.get("asset_name", "ball_valve")]
        self.joint_ids = self.valve.find_joints(VALVE_JOINT)[0]
        self.params = torch.zeros(env.num_envs, len(VALVE_DYNAMICS), device=env.device)
        # The seat-torque action and the observation terms find the values here.
        env.valve_dynamics_term = self

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        ranges: dict[str, tuple[float, float]],
        stuck_prob: float = 0.0,
        stuck_breakaway: tuple[float, float] = STUCK_BREAKAWAY_RANGE,
        asset_name: str = "ball_valve",
    ) -> None:
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        values = torch.empty(len(ids), len(VALVE_DYNAMICS), device=env.device)
        for i, name in enumerate(VALVE_DYNAMICS):
            values[:, i].uniform_(*ranges[name])
        spring_cap = values[:, 0] / SPRING_STATIC_RAD
        values[:, 2] = torch.minimum(values[:, 2], spring_cap)
        stuck = torch.rand(len(ids), device=env.device) < stuck_prob
        if stuck.any():
            values[stuck, 3] = torch.empty(int(stuck.sum()), device=env.device).uniform_(*stuck_breakaway)
            closed = torch.full((int(stuck.sum()), 1), VALVE_CLOSED_RAD, device=env.device)
            self.valve.write_joint_position_to_sim_index(position=closed, joint_ids=self.joint_ids, env_ids=ids[stuck])
            self.valve.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(closed), joint_ids=self.joint_ids, env_ids=ids[stuck]
            )
        self.write(ids, values)

    def write(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Write ``values`` ``(len(env_ids), 5)`` in :data:`VALVE_DYNAMICS` order to the simulator."""
        self.params[env_ids] = values
        joint = self.joint_ids
        self.valve.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=values[:, 0:1].contiguous(),
            joint_viscous_friction_coeff=values[:, 1:2].contiguous(),
            joint_ids=joint,
            env_ids=env_ids,
        )
        self.valve.write_joint_armature_to_sim_index(armature=values[:, 4:5].contiguous(), joint_ids=joint, env_ids=env_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass


def randomize_valve_dynamics_cfg(
    ranges: dict[str, tuple[float, float]] | None = None, stuck_prob: float = STUCK_PROB
) -> EventTermCfg:
    """Reset event sampling the valve dynamics from ``ranges`` (default :data:`VALVE_DYNAMICS_RANGES`)."""
    return EventTermCfg(
        func=randomize_valve_dynamics,
        mode="reset",
        params={"ranges": dict(ranges or VALVE_DYNAMICS_RANGES), "stuck_prob": stuck_prob},
    )


def valve_dynamics_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """The episode's valve dynamics, each divided by its largest sampled value, ``(N, 5)`` (privileged)."""
    term = env.valve_dynamics_term
    scale = term.params.new_tensor(VALVE_DYNAMICS_SCALE)
    return term.params / scale


class ValveLoadAction(ActionTerm):
    """Applies the valve's spring and seat torque (:func:`closing_torque`); takes no actions.

    An action term because action terms are applied every env step before the
    physics runs; the policy's action vector does not change.
    """

    cfg: ValveLoadActionCfg

    def __init__(self, cfg: ValveLoadActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids = self._asset.find_joints(VALVE_JOINT)[0]
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        term = getattr(self._env, "valve_dynamics_term", None)
        if term is None:
            return
        angle = self._asset.data.joint_pos.torch[:, self._joint_ids[0]]
        torque = closing_torque(angle, term.params[:, 2], term.params[:, 3])
        self._asset.set_joint_effort_target_index(target=torque[:, None], joint_ids=self._joint_ids)


@configclass
class ValveLoadActionCfg(ActionTermCfg):
    """Configuration for :class:`ValveLoadAction`."""

    class_type: type = ValveLoadAction
    asset_name: str = "ball_valve"
