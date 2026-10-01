# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the RL ANYmal ball-valve task (teacher/student).

Two observation tiers are built from these terms:

* ``policy`` (the deployable student): arm encoders, gripper motor angle, TCP
  pose from forward kinematics, the previous action, and the *registered*
  valve pose. On hardware the registered pose comes from a one-off board
  registration (AprilTag or touch-probing), so in simulation it is the true
  root pose plus a per-episode bias that stays constant for the episode and a
  small per-step jitter. The student never sees the valve angle; once the
  lever is in the gripper the TCP pose already encodes it.
* ``teacher`` (privileged, simulation only): adds joint velocities, the true
  valve pose, the valve angle and rate, the lever grasp target, and the
  valve-filtered pad contact forces.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import BaseArticulation
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

VALVE_JOINT = "RevoluteJoint"
LEVER_BODY = "alavanca_pivot"
# HiveBoard ball-valve limits are [-pi/2, 0]; negative rotation opens it.
VALVE_CLOSED_RAD = 0.0
VALVE_OPEN_RAD = -1.5707963267948966

# TCP pose in the lever's ``lever_pivot`` frame at the cuRobo expert's grasp,
# measured by ``scripts/rl/measure_grasp_offset.py`` by replaying the expert in
# ``anymal_ball_valve_expert_50.hdf5``. The frame is attached to the lever, so
# the target follows the lever while it turns.
GRASP_OFFSET_POS = (0.0, -0.0244, 0.0)
GRASP_OFFSET_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)


##
# Actions
##


class IntegratedJointPositionAction(JointAction):
    """Joint position target that integrates scaled action increments.

    ``target <- clamp(target + scale * a_f, joint limits)`` with the action
    low-pass filtered, ``a_f <- (1 - smoothing) * a_f + smoothing * action``.
    This is the joint-space version of IndustReal's policy-level action
    integrator: a zero action holds the previous target, so the PD drive keeps
    full stiffness under load (turning the lever) instead of chasing the
    sagging measured position the way :class:`RelativeJointPositionAction`
    does. The filter is part of the action, so the robot runs it too; without
    it the teacher alternated its increments every step and chattered the
    lever at 10 Hz.
    """

    cfg: IntegratedJointPositionActionCfg

    def __init__(self, cfg: IntegratedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._offset = 0.0
        self._fresh = False
        self._filtered = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._target = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._lower = limits[..., 0].clone()
        self._upper = limits[..., 1].clone()

    def apply_actions(self):
        # Newton runs the decimation loop itself, so this is called once per
        # env step there and once per physics step on PhysX. Integrate only on
        # a fresh action so both backends see the same per-step increment.
        if self._fresh:
            alpha = self.cfg.smoothing
            self._filtered.mul_(1.0 - alpha).add_(self.processed_actions, alpha=alpha)
            self._target += self._filtered
            torch.clamp(self._target, self._lower, self._upper, out=self._target)
            self._fresh = False
        self._asset.set_joint_position_target_index(target=self._target, joint_ids=self._joint_ids)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._fresh = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # The action manager resets after the reset events, so the measured
        # joints already hold the randomized reset state: start from there.
        self._target[ids] = self._asset.data.joint_pos.torch[ids][:, self._joint_ids]
        self._filtered[ids] = 0.0
        self._fresh = False


@configclass
class IntegratedJointPositionActionCfg(JointActionCfg):
    """Configuration for :class:`IntegratedJointPositionAction`.

    ``scale`` is the joint increment per env step for a unit action [rad].
    """

    class_type: type = IntegratedJointPositionAction
    smoothing: float = 1.0
    """Weight of the new action in the low-pass filter; 1.0 disables filtering."""


def silence_solver_overflow_warnings(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Startup event: stop MJWarp printing one line per world on solver overflow.

    With thousands of worlds exploring contact-rich states, the per-world
    ``linesearch iterations limit reached`` printf floods stdout (hundreds of
    thousands of lines per minute) and stalls training. The overflow is still
    recorded by the solver; only the device-side printf is disabled. The flag is
    a compile-time kernel constant, so it must be set before the first step.
    """
    from isaaclab_newton.physics import NewtonManager

    solver = getattr(NewtonManager, "_solver", None)
    mjw_model = getattr(solver, "mjw_model", None)
    if mjw_model is None:
        print("[WARN] silence_solver_overflow_warnings: no MJWarp model; warnings stay on.")
        return
    mjw_model.opt.warn_overflow = False


##
# Geometry helpers
##


def _body_frame_w(
    env: ManagerBasedEnv, asset_name: str, body_name: str, offset_pos, offset_rot
) -> tuple[torch.Tensor, torch.Tensor]:
    """World pose [m, xyzw] of a frame rigidly offset from an articulation body.

    Read from the articulation's body poses rather than the FrameTransformer
    sensors: on Newton those are not repopulated after a reset until the next
    physics step, so the first observation of every episode read a zero TCP
    and lever pose (off by the environment origin). The teacher's first actions
    then differed by up to 1.2 between identical states and some environments
    never recovered.
    """
    asset: BaseArticulation = env.scene[asset_name]
    body = asset.find_bodies(body_name)[0][0]
    pos, quat = asset.data.body_pos_w.torch[:, body], asset.data.body_quat_w.torch[:, body]
    off_pos = torch.tensor(offset_pos, device=env.device, dtype=pos.dtype).expand_as(pos)
    off_rot = torch.tensor(offset_rot, device=env.device, dtype=quat.dtype).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, off_pos, off_rot)


def _lever_frame_w(env: ManagerBasedEnv, frame_name: str = "lever_pivot") -> tuple[torch.Tensor, torch.Tensor]:
    """A ``target_frame`` target (e.g. ``lever_pivot``) from the lever body and the scene's offset."""
    frame_cfg = next(f for f in env.scene.cfg.target_frame.target_frames if f.name == frame_name)
    return _body_frame_w(env, "ball_valve", LEVER_BODY, frame_cfg.offset.pos, frame_cfg.offset.rot)


def grasp_target_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert grasp TCP pose on the lever, in world frame [m], xyzw."""
    pos, quat = _lever_frame_w(env)
    off_pos = torch.tensor(GRASP_OFFSET_POS, device=env.device).expand_as(pos)
    off_quat = torch.tensor(GRASP_OFFSET_QUAT, device=env.device).expand_as(quat)
    return math_utils.combine_frame_transforms(pos, quat, off_pos, off_quat)


def tcp_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP pose in world frame [m], xyzw (``ANYMAL_EE``: gripper body plus TCP offset)."""
    from isaaclab_hiveboard.assets import ANYMAL_EE

    return _body_frame_w(env, "robot", ANYMAL_EE.body_name, ANYMAL_EE.tcp_offset.pos, ANYMAL_EE.tcp_offset.rot)


def tcp_grasp_error(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """TCP position [m] and orientation [rad] errors to the grasp target."""
    tcp_pos, tcp_quat = tcp_w(env)
    goal_pos, goal_quat = grasp_target_w(env)
    return torch.norm(tcp_pos - goal_pos, dim=-1), math_utils.quat_error_magnitude(tcp_quat, goal_quat)


def valve_angle(env: ManagerBasedEnv, valve_name: str = "ball_valve") -> torch.Tensor:
    """Valve joint angle [rad], shape ``(num_envs,)``."""
    valve: BaseArticulation = env.scene[valve_name]
    return valve.data.joint_pos.torch[:, valve.find_joints(VALVE_JOINT)[0][0]]


def valve_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """Opening progress in [0, 1]: 0 closed, 1 fully open."""
    q = valve_angle(env)
    return ((q - VALVE_CLOSED_RAD) / (VALVE_OPEN_RAD - VALVE_CLOSED_RAD)).clamp(0.0, 1.0)


def _gripper_closed(env: ManagerBasedEnv) -> torch.Tensor:
    """1 when the gripper action commands close, else 0."""
    return (env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0).float()


##
# Observations
##


def _in_base(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    robot: BaseArticulation = env.scene["robot"]
    pos, quat = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w, quat_w
    )
    return torch.cat((pos, math_utils.quat_unique(quat)), dim=-1)


def tcp_pose_b(env: ManagerBasedEnv) -> torch.Tensor:
    """TCP pose in the base frame from forward kinematics, shape ``(N, 7)``."""
    return _in_base(env, *tcp_w(env))


def valve_pose_b(env: ManagerBasedEnv) -> torch.Tensor:
    """True valve root pose in the base frame (privileged), shape ``(N, 7)``."""
    valve: BaseArticulation = env.scene["ball_valve"]
    return _in_base(env, valve.data.root_pos_w.torch, valve.data.root_quat_w.torch)


def grasp_target_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Moving lever grasp target in the base frame (privileged), shape ``(N, 7)``."""
    return _in_base(env, *grasp_target_w(env))


def tcp_to_grasp_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Grasp target position minus TCP position, in the base frame [m] (privileged)."""
    return grasp_target_b(env)[:, :3] - tcp_pose_b(env)[:, :3]


def valve_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Opening progress and joint rate [rad/s] (privileged), shape ``(N, 2)``."""
    valve: BaseArticulation = env.scene["ball_valve"]
    rate = valve.data.joint_vel.torch[:, valve.find_joints(VALVE_JOINT)[0][0]]
    return torch.stack((valve_progress(env), rate), dim=-1)


def pad_valve_force(env: ManagerBasedEnv, sensor_names: tuple[str, ...] = ("finger_contact", "jaw_contact")):
    """Valve-filtered contact force magnitude on each pad [N] (privileged)."""
    forces = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        matrix = sensor.data.force_matrix_w.torch.reshape(env.num_envs, -1, 3)
        forces.append(torch.norm(matrix.sum(dim=1), dim=-1))
    return torch.stack(forces, dim=-1)


def episode_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time since the episode started, as a fraction of the episode length, ``(N, 1)`` (deployable).

    The robot knows when it started the task. The teacher's references (when to
    close, where the valve should be) are indexed by this time, so without it
    the student has to guess the phase from 5 steps of history.
    """
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl.expert_bank import reference_step

    # Counted from the start of the task, also when an episode starts mid-trajectory.
    return (reference_step(env).float() / env.max_episode_length).unsqueeze(-1)


def gripper_object_detected(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    stopped_below: float = 0.688,
    moved_above: float = 0.3,
    max_speed: float = 0.05,
) -> torch.Tensor:
    """The gripper's "object detected" status, ``(N, 1)`` (deployable; no force sensing).

    Like the Robotiq's gOBJ: commanded closed, and the fingers have come to
    rest (``max_speed`` [rad/s]) short of the closed target, between
    ``moved_above`` and ``stopped_below`` [rad]. Closing on empty air the
    finger joint settles at the 0.70 rad target; on the lever it stops at
    0.667-0.680 rad. ``asset_cfg`` selects the finger joint.
    """
    robot: BaseArticulation = env.scene[asset_cfg.name]
    finger = robot.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    speed = robot.data.joint_vel.torch[:, asset_cfg.joint_ids[0]].abs()
    closing = env.action_manager.get_term("gripper_action").raw_actions[:, 0] < 0.0
    detected = closing & (speed < max_speed) & (finger > moved_above) & (finger < stopped_below)
    return detected.float().unsqueeze(-1)


class registered_valve_b(ManagerTermBase):
    """The valve as a one-off board registration reports it (deployable).

    Registration happens once, at the start of the episode (an AprilTag or a
    touch-probe of the board, plus one look at the lever), so its result is
    latched at reset:

    * the valve root pose with a registration error (``bias_pos`` [m],
      ``bias_rot`` [rad] about each axis), plus ``jitter_pos`` [m] of white
      noise per step;
    * the lever's opening progress at registration, with ``angle_noise``
      [rad] of error. It is *not* updated afterwards: the student has to track
      the lever from its own proprioception once it holds it.

    Returns position (3), unique xyzw quaternion (4) and initial progress (1).
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pose = torch.zeros(env.num_envs, 7, device=env.device)
        self._pose[:, 6] = 1.0
        self._progress = torch.zeros(env.num_envs, 1, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # The observation manager resets after the reset events, so the valve
        # root and joint state read here are the new episode's.
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device)
        )
        if ids.numel() == 0:
            return
        params = self.cfg.params
        n = len(ids)
        pose = valve_pose_b(self._env)[ids]
        bias_pos = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * float(params.get("bias_pos", 0.0))
        euler = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * float(params.get("bias_rot", 0.0))
        bias_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        self._pose[ids, :3] = pose[:, :3] + bias_pos
        self._pose[ids, 3:] = math_utils.quat_unique(math_utils.quat_mul(bias_quat, pose[:, 3:]))
        angle = valve_angle(self._env)[ids]
        angle = angle + (torch.rand(n, device=self.device) * 2.0 - 1.0) * float(params.get("angle_noise", 0.0))
        self._progress[ids, 0] = (angle - VALVE_CLOSED_RAD) / (VALVE_OPEN_RAD - VALVE_CLOSED_RAD)

    def __call__(
        self,
        env: ManagerBasedEnv,
        bias_pos: float = 0.0,
        bias_rot: float = 0.0,
        jitter_pos: float = 0.0,
        angle_noise: float = 0.0,
    ) -> torch.Tensor:
        pose = self._pose.clone()
        pose[:, :3] += torch.randn_like(pose[:, :3]) * jitter_pos
        return torch.cat((pose, self._progress), dim=-1)


##
# Rewards
##


def reach_tcp(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """``1 - tanh(d / std)`` on the TCP-to-grasp distance."""
    dist, _ = tcp_grasp_error(env)
    return 1.0 - torch.tanh(dist / std)


def align_tcp(env: ManagerBasedRLEnv, std: float, gate_dist: float) -> torch.Tensor:
    """Orientation alignment to the grasp, counted only within ``gate_dist`` [m]."""
    dist, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * (dist < gate_dist).float()


def lever_held(env: ManagerBasedEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """True while the gripper is closed at the expert grasp pose on the lever.

    Both position and orientation are required: a closed gripper near the
    lever at the wrong orientation is a fist pushing the lever, which the
    first teacher run learned to do (0.99 s opening, 0% grasps).
    """
    dist, ang = tcp_grasp_error(env)
    return (dist < dist_threshold) & (ang < ang_threshold) & _gripper_closed(env).bool()


def grasp_lever(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed on the lever at the grasp pose."""
    return lever_held(env, dist_threshold, ang_threshold).float()


def align_held(env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Tight orientation alignment [rad] while the lever is held.

    Teacher v3 held the lever at 0.34 rad, just inside the 0.35 rad hold gate,
    so its distilled student fell out of the gate on small imitation errors.
    A narrow kernel while holding pulls the grasp to the expert orientation.
    """
    _, ang = tcp_grasp_error(env)
    return (1.0 - torch.tanh(ang / std)) * lever_held(env, dist_threshold, ang_threshold).float()


def close_far_penalty(env: ManagerBasedRLEnv, dist_threshold: float) -> torch.Tensor:
    """1 while the gripper is closed away from the lever (closing on air)."""
    dist, _ = tcp_grasp_error(env)
    return (dist > dist_threshold).float() * _gripper_closed(env)


def valve_unheld_motion(env: ManagerBasedRLEnv, dist_threshold: float, ang_threshold: float) -> torch.Tensor:
    """Squared valve joint rate [rad^2/s^2] while the lever is *not* held.

    Teacher v2 struck the lever on its approach, flinging it half open at
    ~11 rad/s before grasping. Any valve motion before the grasp is an impact.
    """
    valve: BaseArticulation = env.scene["ball_valve"]
    rate = valve.data.joint_vel.torch[:, valve.find_joints(VALVE_JOINT)[0][0]]
    return rate.square() * (~lever_held(env, dist_threshold, ang_threshold)).float()


def pad_force_excess(env: ManagerBasedRLEnv, max_force: float) -> torch.Tensor:
    """Squared pad-force excess over ``max_force`` [N], summed over both pads."""
    return (pad_valve_force(env) - max_force).clamp(min=0.0).square().sum(dim=-1)


##
# Terminations
##


def valve_open_success(env: ManagerBasedRLEnv, threshold_rad: float) -> torch.Tensor:
    """Success: the valve is within ``threshold_rad`` [rad] of fully open."""
    return valve_angle(env) <= VALVE_OPEN_RAD + threshold_rad


def invalid_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, max_joint_vel: float = 50.0) -> torch.Tensor:
    """Terminate on non-finite state or runaway joint rates [rad/s] (solver blow-ups)."""
    robot: BaseArticulation = env.scene["robot"]
    valve: BaseArticulation = env.scene["ball_valve"]
    arm_vel = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    vel = torch.cat((arm_vel, valve.data.joint_vel.torch), dim=-1)
    nonfinite = (~torch.isfinite(robot.data.joint_pos.torch)).any(dim=-1) | (~torch.isfinite(vel)).any(dim=-1)
    return nonfinite | (vel.abs() > max_joint_vel).any(dim=-1)


##
# Turning-rate command
##


class ValveTurnRateCommand(CommandTerm):
    """Commanded valve turning speed and the reference angle it generates.

    Each episode samples a turning rate [rad/s] from ``rate_range``. The
    reference angle starts at the valve's reset angle and stays there until the
    lever is first held (:func:`lever_held`), so it waits for the grasp instead
    of running away during the reach. From then on it advances toward open at
    that rate whether or not the lever is still held: an earlier version paused
    it whenever the grip drifted out of the hold gate, which froze all tracking
    reward and left the teacher stuck chattering the lever. Rewards track the
    reference, which removes the incentive to open as early as possible and
    makes the turning speed an explicit, operator-set input.

    ``command`` is ``(rate [rad/s], reference progress in [0, 1])``.
    """

    cfg: ValveTurnRateCommandCfg

    def __init__(self, cfg: ValveTurnRateCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rate = torch.zeros(self.num_envs, device=self.device)
        self.ref_angle = torch.zeros(self.num_envs, device=self.device)
        self.engaged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.engaged_time = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_error_rad"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        ref_progress = (self.ref_angle - VALVE_CLOSED_RAD) / (VALVE_OPEN_RAD - VALVE_CLOSED_RAD)
        return torch.stack((self.rate, ref_progress), dim=-1)

    @property
    def reference_done(self) -> torch.Tensor:
        """True once the reference has reached fully open."""
        return self.ref_angle <= VALVE_OPEN_RAD + 1.0e-6

    def _update_metrics(self):
        self.metrics["tracking_error_rad"] = (valve_angle(self._env) - self.ref_angle).abs()

    def _resample_command(self, env_ids: Sequence[int]):
        # The command manager resets after the reset events, so the valve angle
        # read here is the new episode's.
        self.rate[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.rate_range)
        self.ref_angle[env_ids] = valve_angle(self._env)[env_ids]
        self.engaged[env_ids] = False
        self.engaged_time[env_ids] = 0.0

    @property
    def turning(self) -> torch.Tensor:
        """True while the reference is advancing (engaged and not yet open)."""
        return self.engaged & ~self.reference_done

    def _update_command(self):
        self.engaged |= lever_held(self._env, self.cfg.hold_dist, self.cfg.hold_ang)
        self.engaged_time += self._env.step_dt * self.engaged.float()
        # The reference starts once the gripper has had time to close.
        moving = self.engaged_time > self.cfg.engage_delay_s
        # Negative rotation opens the HiveBoard valve.
        self.ref_angle -= self.rate * self._env.step_dt * moving.float()
        self.ref_angle.clamp_(min=VALVE_OPEN_RAD)


@configclass
class ValveTurnRateCommandCfg(CommandTermCfg):
    """Configuration for :class:`ValveTurnRateCommand`."""

    class_type: type = ValveTurnRateCommand
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    rate_range: tuple[float, float] = (0.25, 0.8)
    """Turning rate sampled once per episode [rad/s]."""
    hold_dist: float = 0.03
    """TCP distance [m] within which the lever counts as held (see :func:`lever_held`)."""
    hold_ang: float = 0.35
    """TCP orientation error [rad] within which the lever counts as held."""
    engage_delay_s: float = 0.0
    """Time after the first hold before the reference starts turning [s]. The
    cuRobo expert idles ~0.5 s while its gripper closes; the policy is asked
    to turn right away instead (the turn's joint reference is indexed by valve
    angle, so the expert's pause does not appear in it)."""


def _turn_command(env: ManagerBasedEnv, command_name: str) -> ValveTurnRateCommand:
    return env.command_manager.get_term(command_name)


def turn_rate_command(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Commanded turning rate [rad/s], shape ``(N, 1)`` (deployable: the operator sets it)."""
    return _turn_command(env, command_name).rate.unsqueeze(-1)


def valve_reference_state(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Reference progress and tracking error [rad] (privileged), shape ``(N, 2)``."""
    term = _turn_command(env, command_name)
    error = valve_angle(env) - term.ref_angle
    return torch.stack((term.command[:, 1], error), dim=-1)


def track_valve_reference(
    env: ManagerBasedRLEnv, std: float, dist_threshold: float, ang_threshold: float, command_name: str = "valve_turn"
) -> torch.Tensor:
    """``exp(-(q - q_ref)^2 / std^2)`` while the lever is held; ``std`` [rad]."""
    error = valve_angle(env) - _turn_command(env, command_name).ref_angle
    held = lever_held(env, dist_threshold, ang_threshold).float()
    return torch.exp(-(error / std).square()) * held


def valve_opened_on_schedule(
    env: ManagerBasedRLEnv,
    threshold_rad: float,
    dist_threshold: float,
    ang_threshold: float,
    command_name: str = "valve_turn",
) -> torch.Tensor:
    """1 while the valve is open, held, and the reference has also reached open.

    Opening ahead of the reference earns nothing extra, so there is no reward
    for turning faster than commanded.
    """
    opened = valve_angle(env) <= VALVE_OPEN_RAD + threshold_rad
    on_schedule = _turn_command(env, command_name).reference_done
    return (opened & on_schedule & lever_held(env, dist_threshold, ang_threshold)).float()


def valve_rate_excess(env: ManagerBasedRLEnv, factor: float, command_name: str = "valve_turn") -> torch.Tensor:
    """Squared valve rate [rad/s] above ``factor`` times the commanded rate."""
    valve: BaseArticulation = env.scene["ball_valve"]
    rate = valve.data.joint_vel.torch[:, valve.find_joints(VALVE_JOINT)[0][0]]
    limit = factor * _turn_command(env, command_name).rate
    return (rate.abs() - limit).clamp(min=0.0).square()


def valve_rate_deviation(env: ManagerBasedRLEnv, command_name: str = "valve_turn") -> torch.Tensor:
    """Squared deviation [rad^2/s^2] of the valve rate from the commanded rate while turning.

    Tracking the reference angle alone let the teacher chatter the lever at
    20 Hz around the reference (mean |rate| ~1 rad/s for a 0.3 rad/s command);
    this asks for a steady turn at the commanded speed.
    """
    term = _turn_command(env, command_name)
    valve: BaseArticulation = env.scene["ball_valve"]
    rate = valve.data.joint_vel.torch[:, valve.find_joints(VALVE_JOINT)[0][0]]
    # Opening is negative rotation.
    return (rate + term.rate).square() * term.turning.float()
