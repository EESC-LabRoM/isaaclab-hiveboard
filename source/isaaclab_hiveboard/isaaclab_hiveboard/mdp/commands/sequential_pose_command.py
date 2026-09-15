from __future__ import annotations

import math
import time
from dataclasses import MISSING
from typing import Sequence, Type

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import BaseArticulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm
from isaaclab.managers.manager_term_cfg import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.events import (
    canonicalize_ee_orientation_upward,
)

CUROBO_PATH_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "path": sim_utils.SphereCfg(
            radius=0.008,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
        ),
        "next": sim_utils.SphereCfg(
            radius=0.018,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.3)),
        ),
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        ),
    }
)
"""Plan path: yellow spheres per waypoint, green sphere for the next goal.

RGB frames on sampled waypoints (plus the next goal) show the planned
orientation: red +X is the TCP approach axis, blue +Z is jaw-up.
"""


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Reorder Isaac Lab (x, y, z, w) quaternions to cuRobo (w, x, y, z)."""
    return q[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Reorder cuRobo (w, x, y, z) quaternions to Isaac Lab (x, y, z, w)."""
    return q[..., [1, 2, 3, 0]]


def _pose_str(p: torch.Tensor, q: torch.Tensor) -> str:
    """Compact position + RPY (deg) rendering for plan audit logs."""
    rpy = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(q), dim=-1))
    return (
        f"pos={[round(float(v), 4) for v in p[0].detach().cpu().tolist()]} "
        f"rpy={[round(float(v), 1) for v in rpy[0].detach().cpu().tolist()]}"
    )


class SequentialPoseCommand(CommandTerm):
    """A command term that executes a sequence of commands for a robot's end-effector."""

    cfg: "SequentialPoseCommandCfg"

    def __init__(self, cfg: "SequentialPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self._asset: BaseArticulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if not body_ids:
            raise ValueError(f"Body with name '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'.")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # -- command sequence
        self._current_command_idx = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.long)
        self._command_handlers = [cmd.class_type(cmd, self) for cmd in self.cfg.commands]

        # -- cuRobo solver cache: planner/retargeter construction (kinematics,
        # IK/trajopt solvers, PRM) costs seconds, while solves are sub-second
        # and warm-start from the previous call. Keyed by build config, shared
        # by all handlers so each solver is built once per process lifetime.
        self._curobo_solver_cache: dict = {}

        # -- stuck-segment diagnostics (watchdog + transition log)
        self._seg_time = torch.zeros(self._env.num_envs, device=self._env.device)
        self._seg_warned = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.bool)
        self._prev_command_idx = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.long)

        # -- build default command, quat can't be 0s
        self._command = torch.zeros((self._env.num_envs, 8), device=self._env.device, dtype=torch.float32)
        self._command[:, 0] = 1  # Close gripper
        self._command[:, 7] = 1.0  # xyzw identity (0,0,0,1)

        # -- convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        self._initialize_joint_output()

        # -- optional valve task state
        self.valve_task_goal = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_joint_start = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_joint_des = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.valve_rotate_angle_rad = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._valve_asset: BaseArticulation | None = None
        self._valve_joint_idx: int | None = None
        self._initialize_valve_task()

        # -- optional revolute/prismatic screw coupling
        self._screw_asset: BaseArticulation | None = None
        self._screw_revolute_idx: int | None = None
        self._screw_prismatic_idx: int | None = None
        self._screw_prev_angle = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._screw_axial_target = torch.zeros_like(self._screw_prev_angle)
        self._screw_revolute_target = torch.zeros_like(self._screw_prev_angle)
        self._initialize_screw_coupling()

        if self.cfg.debug_vis:
            self._target_pos_b = torch.zeros((self._env.num_envs, 3), device=self._env.device, dtype=torch.float32)
            self._target_quat_b = torch.zeros((self._env.num_envs, 4), device=self._env.device, dtype=torch.float32)
            self._target_quat_b[:, 3] = 1.0  # xyzw identity (0,0,0,1)

        self._path_markers_visible = False

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired command.

        Default (TCP pose, 8-D):
        - gripper status (positive open, negative close)
        - target end-effector position in the base frame
        - target end-effector orientation (xyzw) in the base frame

        With ``output_joint_positions`` (action positions, 1 + n joints):
        - cuRobo joint waypoints in ``robot_joint_names`` order
        - gripper status (positive open, negative close)
        """
        if self._joint_command is not None:
            return self._joint_command
        return self._command

    def _resample_command(self, env_ids: Sequence[int] | slice | None | torch.Tensor = None):
        """Resets the command sequence for the specified environments."""
        from isaaclab_hiveboard.utils.frame_sensors import refresh_frame_sensors

        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_screw_coupling(env_ids)
        self._sample_valve_task(env_ids)
        refresh_frame_sensors(self._env)
        self._current_command_idx[env_ids] = 0
        self._seg_time[env_ids] = 0.0
        self._seg_warned[env_ids] = False
        self._prev_command_idx[env_ids] = 0
        for handler in self._command_handlers:
            handler.reset(env_ids)
        if self._joint_command is not None:
            self._write_joint_command(self._asset.data.joint_pos.torch[:, self._ik_joint_ids])
        # Isaac Lab's reset() observes the command *before* the first
        # command_manager.compute(). Without this, play.py would step the
        # uninitialized buffer (TCP at the origin) and pull the arm off the
        # lever on the first decimation.
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids_t] = True
        for i, handler in enumerate(self._command_handlers):
            active = env_mask & (self._current_command_idx == i)
            if torch.any(active):
                self._command[active] = handler.update(active)
                if self.cfg.debug_vis:
                    target_pos, target_quat = handler.get_target_in_base_frame(torch.where(active)[0])
                    self._target_pos_b[active] = target_pos
                    self._target_quat_b[active] = target_quat
        self._sync_joint_command()

    def _initialize_joint_output(self) -> None:
        """Publish cuRobo joint waypoints as the command when requested."""
        self._joint_command = None
        self._ik_joint_ids = None
        if not self.cfg.output_joint_positions:
            return
        self._ik_joint_ids = self._resolve_ik_joint_ids()
        self._joint_command = torch.zeros(
            self.num_envs, len(self._ik_joint_ids) + 1, device=self.device, dtype=torch.float32
        )
        self._write_joint_command(self._asset.data.joint_pos.torch[:, self._ik_joint_ids])

    def _resolve_ik_joint_ids(self) -> list[int]:
        names = self.cfg.ik_joint_names
        if not names:
            for cmd in self.cfg.commands:
                names = getattr(cmd, "robot_joint_names", None)
                if names:
                    break
        if not names:
            raise ValueError(
                "output_joint_positions requires ik_joint_names or a cuRobo command with robot_joint_names"
            )
        joint_ids, joint_names = self._asset.find_joints(list(names), preserve_order=True)
        if len(joint_ids) != len(names):
            raise ValueError(f"Expected {len(names)} IK joints {list(names)}, found {len(joint_ids)}: {joint_names}")
        return list(joint_ids)

    def _write_joint_command(self, arm_pos: torch.Tensor) -> None:
        self._joint_command[:, :-1] = arm_pos
        self._joint_command[:, -1] = self._command[:, 0]

    def _sync_joint_command(self) -> None:
        """Copy cuRobo q_des into the published command; otherwise hold the last joints."""
        if self._joint_command is None:
            return
        active, q_des = self.get_curobo_joint_targets()
        if q_des.shape[1] != 0 and torch.any(active):
            self._joint_command[active, :-1] = q_des[active]
        self._joint_command[:, -1] = self._command[:, 0]

    def _initialize_valve_task(self) -> None:
        """Resolve and validate the optional valve articulation and task limits."""
        if self.cfg.valve_asset_name is None:
            return
        if self.cfg.valve_asset_name not in self._env.scene.keys():
            raise ValueError(f"Valve asset '{self.cfg.valve_asset_name}' not found in scene.")
        if not 0.0 <= self.cfg.open_task_prob <= 1.0:
            raise ValueError("open_task_prob must be in [0, 1]")
        valve_span = abs(self.cfg.valve_joint_open - self.cfg.valve_joint_closed)
        if valve_span <= 0.0:
            raise ValueError("valve_joint_open and valve_joint_closed must differ")
        if not 0.0 <= self.cfg.valve_min_delta_rad <= valve_span:
            raise ValueError("valve_min_delta_rad must be within the valve range")

        self._valve_asset = self._env.scene[self.cfg.valve_asset_name]
        joint_ids, _ = self._valve_asset.find_joints(self.cfg.valve_joint_name)
        if len(joint_ids) != 1:
            raise ValueError(
                f"Expected exactly one valve joint named '{self.cfg.valve_joint_name}' "
                f"in '{self.cfg.valve_asset_name}', found {len(joint_ids)}."
            )
        self._valve_joint_idx = joint_ids[0]

    def _sample_valve_task(self, env_ids: torch.Tensor) -> None:
        """Choose a feasible open/close endpoint from the reset valve state.

        The reset event owns valve-state sampling because it also computes the
        matching Spot arm pose. This command term reads that state, samples the
        task direction, and flips infeasible directions near an endpoint so the
        requested motion always satisfies ``valve_min_delta_rad``.
        """
        if self._valve_asset is None or self._valve_joint_idx is None:
            return

        q_start = self._valve_asset.data.joint_pos.torch[env_ids, self._valve_joint_idx]
        q_open = torch.full_like(q_start, self.cfg.valve_joint_open)
        q_closed = torch.full_like(q_start, self.cfg.valve_joint_closed)
        min_delta = float(self.cfg.valve_min_delta_rad)
        can_open = torch.abs(q_open - q_start) >= min_delta
        can_close = torch.abs(q_closed - q_start) >= min_delta
        if torch.any(~(can_open | can_close)):
            bad_q = q_start[~(can_open | can_close)].detach().cpu().tolist()
            raise RuntimeError(
                f"Reset valve states leave no endpoint satisfying valve_min_delta_rad={min_delta}: {bad_q}"
            )

        open_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.open_task_prob
        open_mask = torch.where(can_open & ~can_close, True, open_mask)
        open_mask = torch.where(can_close & ~can_open, False, open_mask)
        goal = torch.where(open_mask, torch.ones_like(q_start), -torch.ones_like(q_start))
        q_des = torch.where(open_mask, q_open, q_closed)

        self.valve_task_goal[env_ids] = goal
        self.valve_joint_start[env_ids] = q_start
        self.valve_joint_des[env_ids] = q_des
        self.recompute_valve_rotate_angle(env_ids)

    def _initialize_screw_coupling(self) -> None:
        """Resolve the optional revolute-to-prismatic screw mechanism."""
        coupling = self.cfg.screw_coupling
        if coupling is None:
            return
        if coupling.asset_name not in self._env.scene.keys():
            raise ValueError(f"Screw asset '{coupling.asset_name}' not found in the scene.")
        if coupling.pitch_m_per_revolution == 0.0:
            raise ValueError("Screw pitch must be non-zero")
        if coupling.lower_limit >= coupling.upper_limit:
            raise ValueError("Screw lower_limit must be less than upper_limit")
        if coupling.velocity_epsilon <= 0.0:
            raise ValueError("Screw velocity_epsilon must be positive")

        self._screw_asset = self._env.scene[coupling.asset_name]
        revolute_ids, _ = self._screw_asset.find_joints(coupling.revolute_joint_name)
        prismatic_ids, _ = self._screw_asset.find_joints(coupling.prismatic_joint_name)
        if len(revolute_ids) != 1 or len(prismatic_ids) != 1:
            raise ValueError("Screw coupling requires exactly one revolute and one prismatic joint")
        self._screw_revolute_idx = revolute_ids[0]
        self._screw_prismatic_idx = prismatic_ids[0]

    def _reset_screw_coupling(self, env_ids: torch.Tensor) -> None:
        """Synchronize coupling state with the articulation after an episode reset."""
        if self._screw_asset is None or self._screw_revolute_idx is None or self._screw_prismatic_idx is None:
            return
        self._screw_prev_angle[env_ids] = self._screw_asset.data.joint_pos.torch[env_ids, self._screw_revolute_idx]
        self._screw_revolute_target[env_ids] = self._screw_prev_angle[env_ids]
        self._screw_axial_target[env_ids] = self._screw_asset.data.joint_pos.torch[env_ids, self._screw_prismatic_idx]
        self._write_screw_targets(env_ids)

    def _write_screw_targets(self, env_ids: torch.Tensor) -> None:
        """Apply the coupled linear target and USD-equivalent resistance torque."""
        if (
            self._screw_asset is None
            or self._screw_revolute_idx is None
            or self._screw_prismatic_idx is None
            or self.cfg.screw_coupling is None
        ):
            return
        coupling = self.cfg.screw_coupling
        velocity = self._screw_asset.data.joint_vel.torch[env_ids, self._screw_revolute_idx]
        speed = torch.abs(velocity)
        coulomb = torch.where(
            speed > coupling.velocity_epsilon,
            -coupling.coulomb_friction * torch.sign(velocity),
            torch.zeros_like(velocity),
        )
        stiction = torch.where(
            speed < coupling.velocity_epsilon,
            -coupling.stiction * velocity / coupling.velocity_epsilon,
            torch.zeros_like(velocity),
        )
        friction = -coupling.viscous_friction * velocity + coulomb + stiction

        distance_to_limit = torch.minimum(
            self._screw_axial_target[env_ids] - coupling.lower_limit,
            coupling.upper_limit - self._screw_axial_target[env_ids],
        )
        end_damping = torch.where(
            distance_to_limit < coupling.end_stop_activation_distance,
            coupling.end_stop_base_damping
            + (coupling.end_stop_scale / (distance_to_limit.clamp_min(0.0) + 1.0e-3)) ** coupling.end_stop_power,
            torch.zeros_like(distance_to_limit),
        )
        resistance = friction - end_damping * velocity

        self._screw_asset.set_joint_position_target(
            self._screw_axial_target[env_ids, None],
            joint_ids=[self._screw_prismatic_idx],
            env_ids=env_ids,
        )
        if coupling.command_joint_angle_scale != 0.0:
            # Drive the revolute with a position target so the bulb turns with
            # the commanded screw. Do not write_joint_state_to_sim: a kinematic
            # teleport while the gripper is in contact explodes PhysX.
            self._screw_asset.set_joint_position_target(
                self._screw_revolute_target[env_ids, None],
                joint_ids=[self._screw_revolute_idx],
                env_ids=env_ids,
            )
        else:
            self._screw_asset.set_joint_effort_target(
                resistance[:, None],
                joint_ids=[self._screw_revolute_idx],
                env_ids=env_ids,
            )

    def _update_screw_coupling(self) -> None:
        """Convert measured revolute travel into axial travel at the screw pitch."""
        if self._screw_asset is None or self._screw_revolute_idx is None or self.cfg.screw_coupling is None:
            return
        coupling = self.cfg.screw_coupling
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        angle = self._screw_asset.data.joint_pos.torch[:, self._screw_revolute_idx]
        raw_delta = angle - self._screw_prev_angle
        angle_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
        self._screw_axial_target += angle_delta * coupling.pitch_m_per_revolution / (2.0 * math.pi)
        self._screw_axial_target.clamp_(min=coupling.lower_limit, max=coupling.upper_limit)
        self._screw_prev_angle.copy_(angle)
        self._write_screw_targets(env_ids)

    def recompute_valve_rotate_angle(self, env_ids: torch.Tensor) -> None:
        """Update the EE arc angle from the valve's remaining joint error."""
        if self._valve_asset is None or self._valve_joint_idx is None:
            return
        q_current = self._valve_asset.data.joint_pos.torch[env_ids, self._valve_joint_idx]
        self.valve_rotate_angle_rad[env_ids] = (self.valve_joint_des[env_ids] - q_current) * float(
            self.cfg.valve_ee_joint_angle_scale
        )

    def _handler_label(self, idx: int) -> str:
        if idx >= len(self._command_handlers):
            return "done"
        return type(self._command_handlers[idx]).__name__

    def _dump_stall(self, env_index: int) -> None:
        """Print everything needed to see why segment ``env_index`` is stuck."""
        idx = int(self._current_command_idx[env_index].item())
        handler = self._command_handlers[idx]
        print(
            f"[STALL] env={env_index} stuck on seg {idx}/{len(self._command_handlers)} "
            f"({type(handler).__name__}) for >={float(self.cfg.stall_timeout_s):.1f}s "
            f"(seg_time={float(self._seg_time[env_index].item()):.1f}s)",
            flush=True,
        )
        env_ids = torch.tensor([env_index], device=self.device, dtype=torch.long)
        try:
            target_pos_b, target_quat_b = handler.get_target_in_base_frame(env_ids)
            ee_pos_b, ee_quat_b = self._get_ee_in_base_frame(env_ids)
            pos_err = float(torch.linalg.vector_norm(ee_pos_b[0] - target_pos_b[0]).item())
            ori_err = float(math.degrees(math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)[0].item()))
            print(
                f"[STALL] target_pos_b={target_pos_b[0].detach().cpu().tolist()} "
                f"ee_pos_b={ee_pos_b[0].detach().cpu().tolist()} pos_err_m={pos_err:.4f} "
                f"ori_err_deg={ori_err:.2f} gripper_cmd={float(self._command[env_index, 0].item()):+.1f}",
                flush=True,
            )
        except Exception as err:  # Gripper holds have no Cartesian target
            print(f"[STALL] (no Cartesian target readout: {err})", flush=True)
        if isinstance(handler, _CuroboPlannedGoToFrameHandler):
            if handler._fallback:
                print("[STALL] curobo: plan failed, running fallback direct servo", flush=True)
            elif handler._waypoint_pos_b is None:
                print("[STALL] curobo: waypoints not planned yet", flush=True)
            else:
                wi = int(handler._waypoint_index[env_ids].item())
                total = int(handler._waypoint_pos_b.shape[1])
                print(f"[STALL] curobo: waypoint {wi}/{total}", flush=True)
        try:
            q = self._asset.data.joint_pos[env_index].detach().cpu().tolist()
            print(f"[STALL] joint_pos={[round(v, 3) for v in q]}", flush=True)
        except Exception as err:
            print(f"[STALL] (no joint readout: {err})", flush=True)
        if self._valve_asset is not None and self._valve_joint_idx is not None:
            qv = float(self._valve_asset.data.joint_pos[env_index, self._valve_joint_idx].item())
            qd = float(self.valve_joint_des[env_index].item())
            print(f"[STALL] valve_q={qv:.4f} valve_des={qd:.4f}", flush=True)

    def _update_command(self):
        """Updates the command based on the current state of the command sequence."""
        step_dt = float(getattr(self._env, "step_dt", 0.02))
        changed = self._current_command_idx != self._prev_command_idx
        if torch.any(changed):
            if self.cfg.log_transitions:
                for e in torch.where(changed)[0].tolist():
                    prev = int(self._prev_command_idx[e].item())
                    new = int(self._current_command_idx[e].item())
                    print(
                        f"[SEQ] env={e} seg {prev} ({self._handler_label(prev)}) done "
                        f"in {float(self._seg_time[e].item()):.2f}s "
                        f"-> seg {new} ({self._handler_label(new)})",
                        flush=True,
                    )
            self._seg_time[changed] = 0.0
            self._seg_warned[changed] = False
            self._prev_command_idx[changed] = self._current_command_idx[changed]
        self._seg_time += step_dt
        if float(self.cfg.stall_timeout_s) > 0.0:
            stalled = (
                (self._seg_time >= float(self.cfg.stall_timeout_s))
                & ~self._seg_warned
                & (self._current_command_idx < len(self._command_handlers))
            )
            if torch.any(stalled):
                for e in torch.where(stalled)[0].tolist():
                    self._dump_stall(e)
                self._seg_warned[stalled] = True
        # Update the command from the current handler
        for i, handler in enumerate(self._command_handlers):
            env_mask = self._current_command_idx == i

            if not torch.any(env_mask):
                continue

            # check if the current command is done
            env_ids = torch.where(env_mask)[0]
            are_done = handler.is_done(env_ids)
            done_env_ids = env_ids[are_done]

            if len(done_env_ids) > 0:
                self._current_command_idx[done_env_ids] = self._current_command_idx[done_env_ids] + 1
                if i < (len(self._command_handlers) - 1):
                    self._command_handlers[i + 1].reset(done_env_ids)

            # update the command for the current handler
            self._command[env_mask] = handler.update(env_mask)

            if self.cfg.debug_vis:
                target_pos_b, target_quat_b = handler.get_target_in_base_frame(torch.where(env_mask)[0])
                self._target_pos_b[env_mask] = target_pos_b
                self._target_quat_b[env_mask] = target_quat_b

        # Apply after handlers so ScrewFrame can publish this step's revolute target.
        self._update_screw_coupling()
        self._sync_joint_command()

    def is_done(self) -> torch.Tensor:
        """Check if all commands were finished."""
        return self._current_command_idx == len(self._command_handlers)

    def get_curobo_joint_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return direct joint targets for environments executing a cuRobo term.

        Used when ``output_joint_positions`` is set, and by
        :class:`~isaaclab_hiveboard.mdp.actions.curobo_joint_action.CuroboJointPositionAction`.
        Inactive or unplanned segments leave the last joints unchanged.
        """
        targets = torch.zeros(self.num_envs, 0, device=self.device, dtype=self._command.dtype)
        active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for command_idx, handler in enumerate(self._command_handlers):
            if not isinstance(handler, _CuroboPlannedGoToFrameHandler):
                continue
            if handler._joint_target is None:
                continue
            if targets.shape[1] == 0:
                targets = torch.zeros(
                    self.num_envs,
                    handler._joint_target.shape[1],
                    device=self.device,
                    dtype=handler._joint_target.dtype,
                )
            elif targets.shape[1] != handler._joint_target.shape[1]:
                raise RuntimeError("All cuRobo command terms must command the same joint count.")
            env_mask = self._current_command_idx == command_idx
            targets[env_mask] = handler._joint_target[env_mask]
            active |= env_mask
        return active, targets

    def _update_metrics(self):
        """This command term does not have any metrics to update."""
        pass

    def get_curobo_solver(self, key: tuple, make):
        """Return a cached cuRobo solver, building it once per config.

        Args:
            key: Hashable build config identifying the solver.
            make: Zero-argument factory called on cache miss. The caller must
                invoke it inside ``torch.inference_mode(False)`` (and the
                cuRobo warp guard), like a fresh construction.

        Returns:
            The cached solver and the wall-clock build time in seconds
            (0.0 on a cache hit).
        """
        solver = self._curobo_solver_cache.get(key)
        if solver is not None:
            return solver, 0.0
        start = time.perf_counter()
        solver = make()
        build_s = time.perf_counter() - start
        self._curobo_solver_cache[key] = solver
        return solver, build_s

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)

            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)

            if self.cfg.path_debug_vis and not hasattr(self, "curobo_path_visualizer"):
                self.curobo_path_visualizer = VisualizationMarkers(
                    CUROBO_PATH_MARKER_CFG.replace(prim_path="/Visuals/Command/curobo_path")
                )
            if hasattr(self, "curobo_path_visualizer"):
                self.curobo_path_visualizer.set_visibility(self.cfg.path_debug_vis)
                self._path_markers_visible = False

        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "curobo_path_visualizer"):
                self.curobo_path_visualizer.set_visibility(False)
                self._path_markers_visible = False

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        asset_pos = self._asset.data.root_pos_w.torch
        asset_quat = self._asset.data.root_quat_w.torch

        # goal end-effector pose
        target_pos_w, target_quat_w = math_utils.combine_frame_transforms(
            asset_pos,
            asset_quat,
            self._target_pos_b,
            self._target_quat_b,
        )
        self.goal_pose_visualizer.visualize(target_pos_w, target_quat_w)

        # current end-effector pose
        ee_pos_b, ee_quat_b = self._get_ee_in_world_frame(slice(None))
        self.current_pose_visualizer.visualize(ee_pos_b, ee_quat_b)

        # Active command path (env 0): waypoint spheres + next goal
        if self.cfg.path_debug_vis and hasattr(self, "curobo_path_visualizer"):
            self._update_curobo_path_vis()

    def _update_curobo_path_vis(self, env_index: int = 0) -> None:
        """Show an actual cuRobo plan or the active direct command's geometry."""
        from isaaclab_hiveboard.utils.command_path import active_command_path

        show = False
        if int(self._current_command_idx[env_index].item()) < len(self._command_handlers):
            handler = self._command_handlers[int(self._current_command_idx[env_index].item())]
            path = active_command_path(handler, self._command, env_index)
            if path is not None:
                env_ids = torch.tensor([env_index], device=self.device, dtype=torch.long)
                pos_b, quat_b, next_idx = path
                n = pos_b.shape[0]
                root_pos = self._asset.data.root_pos_w.torch[env_ids].expand(n, -1)
                root_quat = self._asset.data.root_quat_w.torch[env_ids].expand(n, -1)
                pos_w, quat_w = math_utils.combine_frame_transforms(root_pos, root_quat, pos_b, quat_b)
                # Frames on ~10 sampled waypoints (always including the last)
                # plus the next goal, so orientation twists are visible.
                step = max(1, n // 10)
                frame_idx = list(range(0, n, step))
                if frame_idx[-1] != n - 1:
                    frame_idx.append(n - 1)
                if next_idx not in frame_idx:
                    frame_idx.append(next_idx)
                n_frames = len(frame_idx)
                identity = torch.zeros(n + 1, 4, device=self.device)
                identity[:, 3] = 1.0
                translations = torch.cat(
                    [pos_w, pos_w[next_idx : next_idx + 1], pos_w[frame_idx], pos_w[next_idx : next_idx + 1]],
                    dim=0,
                )
                orientations = torch.cat(
                    [identity, quat_w[frame_idx], quat_w[next_idx : next_idx + 1]],
                    dim=0,
                )
                marker_indices = translations.new_zeros(n + n_frames + 2, dtype=torch.long)
                marker_indices[n] = 1
                marker_indices[n + 1 :] = 2
                self.curobo_path_visualizer.set_visibility(True)
                self.curobo_path_visualizer.visualize(
                    translations, orientations, marker_indices=marker_indices
                )
                self._path_markers_visible = True
                show = True
        if not show and self._path_markers_visible:
            self.curobo_path_visualizer.set_visibility(False)
            self._path_markers_visible = False

    def _body_pose_in_base(self, env_ids: torch.Tensor | slice):
        """Commanded body (``arm_link_wr1``) pose in the robot base frame."""
        return math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            self._asset.data.body_pos_w.torch[env_ids, self._body_idx],
            self._asset.data.body_quat_w.torch[env_ids, self._body_idx],
        )

    def _body_pose_to_tcp_pose(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | slice):
        """Compose the flange→TCP offset onto a body pose."""
        if self._offset_pos is None or self._offset_rot is None:
            return pos, quat
        return math_utils.combine_frame_transforms(
            pos, quat, self._offset_pos[env_ids], self._offset_rot[env_ids]
        )

    def _tcp_pose_to_body_pose(self, pos: torch.Tensor, quat: torch.Tensor, env_ids: torch.Tensor | slice):
        """Map a TCP pose back to the flange body with the inverse offset."""
        if self._offset_pos is None or self._offset_rot is None:
            return pos, quat
        offset_pos = self._offset_pos[env_ids]
        offset_rot = self._offset_rot[env_ids]
        if offset_pos.shape[0] != pos.shape[0]:
            offset_pos = offset_pos.expand(pos.shape[0], -1)
            offset_rot = offset_rot.expand(pos.shape[0], -1)
        body_from_tcp_quat = math_utils.quat_inv(offset_rot)
        body_from_tcp_pos = -math_utils.quat_apply(body_from_tcp_quat, offset_pos)
        return math_utils.combine_frame_transforms(pos, quat, body_from_tcp_pos, body_from_tcp_quat)

    def _get_ee_in_base_frame(self, env_ids: torch.Tensor | slice):
        """TCP pose in the robot base frame."""
        return self._body_pose_to_tcp_pose(*self._body_pose_in_base(env_ids), env_ids)

    def _get_ee_in_world_frame(self, env_ids: torch.Tensor | slice):
        """TCP pose in the world frame."""
        return self._body_pose_to_tcp_pose(
            self._asset.data.body_pos_w.torch[env_ids, self._body_idx],
            self._asset.data.body_quat_w.torch[env_ids, self._body_idx],
            env_ids,
        )


class _BaseCmdHandler:
    """Base class for command handlers."""

    def __init__(self, cfg: BaseCmd, command_term: SequentialPoseCommand):
        self.cfg = cfg
        self._command_term = command_term
        self._asset = command_term._asset
        self._device = command_term.device
        self._num_envs = command_term.num_envs
        self._dt = float(command_term._env.step_dt)

    def reset(self, env_ids: torch.Tensor):
        """Reset the handler for the given environment IDs."""
        raise NotImplementedError

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        """Update the command for the given environment IDs."""
        raise NotImplementedError

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check if the command is done for the given environment IDs."""
        raise NotImplementedError

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        """Get the target pose in the base frame for the given environment IDs."""
        raise NotImplementedError

    def _pack_command(self, gripper_open: bool, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        command = torch.zeros(pos.shape[0], 8, device=self._device)
        command[:, 0] = 1.0 if gripper_open else -1.0
        command[:, 1:4] = pos
        command[:, 4:8] = quat
        return command

    def _apply_target_offset(self, pos: torch.Tensor, quat: torch.Tensor):
        """Compose the authored goal/pivot offset in its reference frame."""
        return math_utils.combine_frame_transforms(
            pos, quat,
            pos.new_tensor(self.cfg.target_offset_pos).expand_as(pos),
            quat.new_tensor(self.cfg.target_offset_rot).expand_as(quat),
        )

    def _step_pos_towards(
        self,
        current: torch.Tensor,
        target: torch.Tensor,
        velocity: float,
    ) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``velocity * dt``."""
        delta = target - current
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(velocity * self._dt / dist.clamp(min=1.0e-8), max=1.0)
        return current + delta * scale

    def _step_quat_towards(self, current: torch.Tensor, target: torch.Tensor, angular_velocity: float) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``angular_velocity * dt``."""
        err = math_utils.quat_box_minus(target, current)
        angle = torch.linalg.vector_norm(err, dim=-1, keepdim=True)
        scale = torch.clamp(angular_velocity * self._dt / angle.clamp(min=1.0e-8), max=1.0)
        return math_utils.quat_box_plus(current, err * scale)


class _GoToFrameHandler(_BaseCmdHandler):
    """Handles the GoToFrame command."""

    def __init__(
        self,
        cfg: GoToFrameCfg,
        command_term: SequentialPoseCommand,
    ):
        super().__init__(cfg, command_term)
        self.cfg: GoToFrameCfg
        if self.cfg.velocity <= 0.0:
            raise ValueError("GoToFrameCfg.velocity must be positive")
        if self.cfg.angular_velocity <= 0.0:
            raise ValueError("GoToFrameCfg.angular_velocity must be positive")
        if self.cfg.distance_threshold < 0.0:
            raise ValueError("GoToFrameCfg.distance_threshold must be non-negative")
        if self.cfg.orientation_threshold_deg < 0.0:
            raise ValueError("GoToFrameCfg.orientation_threshold_deg must be non-negative")

        if cfg.target_position_env is None:
            self._frame = command_term._env.scene[cfg.frame_name]
            self._frame_idx = self._frame.data.target_frame_names.index(cfg.target_frame_name)
        self.command_pos_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.command_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.command_quat_b[:, 3] = 1.0
        self._held_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self._held_quat_b[:, 3] = 1.0
        self._upward_flip = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self._ori_threshold_rad = math.radians(self.cfg.orientation_threshold_deg)

    def reset(self, env_ids: torch.Tensor):
        ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
        self.command_pos_b[env_ids] = ee_pos_b
        self.command_quat_b[env_ids] = ee_quat_b
        self._held_quat_b[env_ids] = ee_quat_b
        self._upward_flip[env_ids] = False
        if (
            self.cfg.canonicalize_upward
            and not self.cfg.hold_current_orientation
            and self.cfg.target_position_env is None
        ):
            # Pick the symmetric grasp once per segment. Re-evaluating "up"
            # every step flips the goal by 180 degrees when contact nudges a
            # horizontal handle across +/-90 degrees, preventing completion.
            _, target_quat_b = self.get_target_in_base_frame(env_ids)
            upward_quat_b = canonicalize_ee_orientation_upward(target_quat_b)
            self._upward_flip[env_ids] = torch.abs(torch.sum(target_quat_b * upward_quat_b, dim=-1)) < 0.5

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)

        self.command_pos_b[env_ids] = self._step_pos_towards(
            self.command_pos_b[env_ids], target_pos_b, self.cfg.velocity
        )
        self.command_quat_b[env_ids] = self._step_quat_towards(
            self.command_quat_b[env_ids], target_quat_b, self.cfg.angular_velocity
        )
        return self._pack_command(
            self.cfg.gripper_open,
            self.command_pos_b[env_ids],
            self.command_quat_b[env_ids],
        )

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
        target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
        pos_err = torch.linalg.vector_norm(ee_pos_b - target_pos_b, dim=-1)
        ori_err = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)
        done = (pos_err <= self.cfg.distance_threshold) & (ori_err <= self._ori_threshold_rad)
        threshold = getattr(self.cfg, "valve_done_threshold_rad", None)
        term = self._command_term
        if (
            threshold is not None
            and float(threshold) > 0.0
            and term._valve_asset is not None
            and term._valve_joint_idx is not None
        ):
            # Contact-rich arc: the TCP can sit centimeters off while the
            # valve is already turned. Advance on the valve angle instead of
            # hanging on Cartesian thresholds.
            valve_q = term._valve_asset.data.joint_pos.torch[env_ids, term._valve_joint_idx]
            done = done | (torch.abs(valve_q - term.valve_joint_des[env_ids]) <= float(threshold))
        return done

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Target pose in base frame
        if self.cfg.target_position_env is not None:
            target_pos_w = self._command_term._env.scene.env_origins[env_ids] + torch.tensor(
                self.cfg.target_position_env, device=self._device
            )
            target_quat_w = None
            if self.cfg.target_orientation_env is not None:
                target_quat_w = torch.tensor(self.cfg.target_orientation_env, device=self._device).expand(
                    len(env_ids), -1
                )
            target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w.torch[env_ids],
                self._asset.data.root_quat_w.torch[env_ids],
                target_pos_w,
                target_quat_w,
            )
            return target_pos_b, (self._held_quat_b[env_ids] if target_quat_w is None else target_quat_b)
        target_pos_w = self._frame.data.target_pos_w.torch[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w.torch[env_ids, self._frame_idx]

        target_pos_w, target_quat_w = self._apply_target_offset(target_pos_w, target_quat_w)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            target_pos_w,
            target_quat_w,
        )
        if self.cfg.canonicalize_upward:
            flip_x = target_quat_b.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(target_quat_b)
            target_quat_b = torch.where(
                self._upward_flip[env_ids, None], math_utils.quat_mul(target_quat_b, flip_x), target_quat_b
            )
        if self.cfg.hold_current_orientation:
            target_quat_b = self._held_quat_b[env_ids]
        if self.cfg.position_override_b is not None:
            for axis, value in enumerate(self.cfg.position_override_b):
                if value is not None:
                    target_pos_b[:, axis] = float(value)

        return target_pos_b, target_quat_b


class _CuroboPlannedGoToFrameHandler(_GoToFrameHandler):
    """Follow cuRobo joint-plan waypoints for a constrained Cartesian move."""

    def __init__(self, cfg: "CuroboPlannedGoToFrameCfg", command_term: SequentialPoseCommand):
        super().__init__(cfg, command_term)
        self.cfg: CuroboPlannedGoToFrameCfg
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        # Scalar is sufficient: like the synchronized-env check below, all
        # envs plan each segment together.
        self._fallback = False

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index[env_ids] = 0
        self._fallback = False
        if len(env_ids) != self._num_envs:
            raise RuntimeError("CuroboPlannedGoToFrameCfg currently requires synchronized environments.")

    def _plan(self, env_ids: torch.Tensor) -> None:
        """Build a bounded cuRobo plan and convert it to TCP pose waypoints."""
        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        with curobo_compatible_warp():
            from curobo.types import DeviceCfg, JointState

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            if self.cfg.reference_pos_env is not None and self.cfg.reference_quat_xyzw is not None:
                self._plan_from_reference(env_ids, robot_cfg)
                return
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import GoalToolPose, Pose

            # Reuse one planner per build config (see _curobo_solver_cache):
            # construction costs seconds, solves are sub-second and warm-start
            # from the previous call. The factory must run in inference_mode
            # (False), otherwise cuRobo's reusable goal buffers become
            # inference tensors that TrajOpt cannot update in place.
            cache_key = (
                "planner",
                self.cfg.robot_curobo_yaml,
                self.cfg.robot_urdf,
                self.cfg.num_ik_seeds,
                self.cfg.num_trajopt_seeds,
                self.cfg.interpolation_buffer_size,
            )

            def _make_planner() -> MotionPlanner:
                with torch.inference_mode(False):
                    return MotionPlanner(
                        MotionPlannerCfg.create(
                            robot=robot_cfg,
                            self_collision_check=False,
                            use_cuda_graph=False,
                            num_ik_seeds=self.cfg.num_ik_seeds,
                            num_trajopt_seeds=self.cfg.num_trajopt_seeds,
                            interpolation_dt=self._dt,
                            interpolation_buffer_size=self.cfg.interpolation_buffer_size,
                            device_cfg=DeviceCfg(),
                        )
                    )

            planner, build_s = self._command_term.get_curobo_solver(cache_key, _make_planner)
            solve_start = time.perf_counter()
            if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                raise ValueError("CuroboPlannedGoToFrameCfg requires pose_command.body_offset")
            try:
                target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
                flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                    target_pos_b, target_quat_b, env_ids
                )
                # Skip the solver when already there (e.g. a home-to-home leg):
                # publish the current pose/joints as a single waypoint instead
                # of burning seconds of trajopt on a no-op.
                rep = env_ids[0:1]
                ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(rep)
                if float(torch.linalg.vector_norm(ee_pos_b[0] - target_pos_b[0]).item()) <= float(
                    self.cfg.distance_threshold
                ) and float(
                    math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)[0].item()
                ) <= math.radians(
                    self.cfg.orientation_threshold_deg
                ):
                    rep_jids, _ = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
                    rep_q = self._asset.data.joint_pos.torch[rep][:, rep_jids]
                    self._publish_plan(ee_pos_b, ee_quat_b, rep_q.unsqueeze(0), origin="trivial")
                    return
                joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
                current = JointState.from_position(
                    self._asset.data.joint_pos.torch[env_ids][:, joint_ids],
                    joint_names=joint_names,
                )
                # The Isaac articulation root and the FR3 URDF root are not
                # the same frame.  Calibrate the constant transform from the
                # current shared joint configuration, then express the Isaac
                # target flange pose in cuRobo's URDF base frame.
                curobo_flange = planner.compute_kinematics(current).tool_poses.get_link_pose(planner.tool_frames[0])
                isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                    self._asset.data.root_pos_w.torch[env_ids],
                    self._asset.data.root_quat_w.torch[env_ids],
                    self._asset.data.body_pos_w.torch[env_ids, self._command_term._body_idx],
                    self._asset.data.body_quat_w.torch[env_ids, self._command_term._body_idx],
                )
                # Planning audit (first active env): measured start vs
                # commanded goal, flange and TCP, all in the base frame. An
                # inverted axis shows up as a negated component or ~180 deg
                # RPY jump here, before cuRobo is even involved.
                e0 = int(env_ids[0].item())
                start_tcp_b, start_tcp_q = math_utils.combine_frame_transforms(
                    isaac_flange_pos_b[0:1],
                    isaac_flange_quat_b[0:1],
                    self._command_term._offset_pos[env_ids[0:1]],
                    self._command_term._offset_rot[env_ids[0:1]],
                )

                try:
                    audit_seg = self._command_term._command_handlers.index(self)
                except ValueError:
                    audit_seg = "?"
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} start EE-flange_b {_pose_str(isaac_flange_pos_b[0:1], isaac_flange_quat_b[0:1])}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} goal  EE-flange_b {_pose_str(flange_pos_b[0:1], flange_quat_b[0:1])}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} start EE-TCP_b    {_pose_str(start_tcp_b, start_tcp_q)}",
                    flush=True,
                )
                print(
                    f"[INFO] plan seg {audit_seg} env={e0} goal  EE-TCP_b    {_pose_str(target_pos_b[0:1], target_quat_b[0:1])}",
                    flush=True,
                )
                # cuRobo quaternions are (w, x, y, z); math_utils is (x, y, z, w).
                curobo_flange_quat_b = _wxyz_to_xyzw(curobo_flange.quaternion)
                flange_quat_inv_c = math_utils.quat_inv(curobo_flange_quat_b)
                flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
                curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
                    isaac_flange_pos_b,
                    isaac_flange_quat_b,
                    flange_pos_inv_c,
                    flange_quat_inv_c,
                )
                flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                    curobo_base_pos_b,
                    curobo_base_quat_b,
                    flange_pos_b,
                    flange_quat_b,
                )
                # Isaac Lab evaluates command terms under ``inference_mode``.
                # cuRobo's seed IK intentionally differentiates its pose cost;
                # ``enable_grad`` alone cannot override inference mode.  Clone
                # the input tensors in this context too: otherwise cuRobo
                # adopts an inference-mode goal tensor as its mutable cache.
                with torch.inference_mode(False), torch.enable_grad():
                    current = JointState.from_position(current.position.clone(), joint_names=joint_names)
                    goal = GoalToolPose.from_poses(
                        {
                            planner.tool_frames[0]: Pose(
                                position=flange_pos_c.clone(),
                                quaternion=_xyzw_to_wxyz(flange_quat_c.clone()),
                            )
                        },
                        ordered_tool_frames=planner.tool_frames,
                        num_goalset=1,
                    )
                    # NOTE: enable_graph_attempt=99 effectively disables graph
                    # seeding (attempts only run 0..max_plan_attempts-1, which
                    # never reach 99). Pure IK + trajopt is enough here and far
                    # cheaper than PRM queries; lower it only if you also want
                    # graph seeds on later attempts.
                    result = planner.plan_pose(
                        goal,
                        current,
                        max_attempts=self.cfg.max_plan_attempts,
                        enable_graph_attempt=99,
                    )
                if result is None or not bool(result.success.all().item()):
                    try:
                        seg = self._command_term._command_handlers.index(self)
                    except ValueError:
                        seg = "?"
                    print(
                        f"[WARN] cuRobo plan failed on seg {seg}; falling back to direct servo for "
                        f"{self.cfg.target_frame_name or self.cfg.target_position_env}. "
                        f"start_q={current.position.detach().cpu().tolist()} "
                        f"isaac_flange={isaac_flange_pos_b.detach().cpu().tolist()} "
                        f"curobo_flange={curobo_flange.position.detach().cpu().tolist()} "
                        f"curobo_base={curobo_base_pos_b.detach().cpu().tolist()} "
                        f"flange_pos_c={flange_pos_c.detach().cpu().tolist()} "
                        f"flange_quat_c={flange_quat_c.detach().cpu().tolist()}"
                    )
                    # Degrade to the parent velocity-limited servo for this
                    # segment instead of killing the episode. Servo state was
                    # seeded from the current EE pose at reset, and
                    # _joint_target stays None so the action term does not
                    # apply stale joint overrides.
                    self._fallback = True
                    return
                trajectory = result.get_interpolated_plan().position
                if trajectory.ndim == 4:
                    trajectory = trajectory[:, 0]
                poses = planner.compute_kinematics(
                    JointState.from_position(trajectory[0], joint_names=joint_names)
                ).tool_poses.get_link_pose(planner.tool_frames[0])
                tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                    poses.position,
                    _wxyz_to_xyzw(poses.quaternion),
                    self._command_term._offset_pos[0].expand_as(poses.position),
                    self._command_term._offset_rot[0].expand_as(poses.quaternion),
                )
                # Convert visual TCP waypoints back to Isaac's base frame;
                # direct joint execution below is the authoritative motion.
                tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                    curobo_base_pos_b[0].expand_as(tcp_pos),
                    curobo_base_quat_b[0].expand_as(tcp_quat),
                    tcp_pos,
                    tcp_quat,
                )
                solve_s = time.perf_counter() - solve_start
                self._publish_plan(
                    tcp_pos, tcp_quat, trajectory, origin="planned", build_s=build_s, solve_s=solve_s
                )
            except Exception as err:
                # A throwing solver must not kill the episode (the original
                # cuRobo crash mode): degrade to direct servo like a failed
                # plan. Misconfiguration still raises above, outside this try.
                try:
                    seg = self._command_term._command_handlers.index(self)
                except ValueError:
                    seg = "?"
                print(
                    f"[WARN] cuRobo plan threw on seg {seg} "
                    f"({type(err).__name__}: {err}); falling back to direct servo",
                    flush=True,
                )
                self._fallback = True
                return

    def _publish_plan(
        self,
        tcp_pos: torch.Tensor,
        tcp_quat: torch.Tensor,
        trajectory: torch.Tensor,
        origin: str,
        build_s: float = 0.0,
        solve_s: float = 0.0,
    ) -> None:
        """Store waypoints/joints and emit plan diagnostics (shared by both planners)."""
        if getattr(self.cfg, "valve_done_threshold_rad", None) is not None:
            self._validate_arc_radius(tcp_pos)
        self._waypoint_pos_b = tcp_pos.unsqueeze(0)
        self._waypoint_quat_b = tcp_quat.unsqueeze(0)
        self._joint_waypoints = trajectory
        self._joint_target = torch.zeros(
            self._num_envs, trajectory.shape[-1], device=self._device, dtype=trajectory.dtype
        )
        print(
            f"[INFO] cuRobo {origin} {tcp_pos.shape[0]} TCP waypoints for "
            f"{self.cfg.target_frame_name or self.cfg.target_position_env} "
            f"(build={build_s:.2f}s solve={solve_s:.2f}s)."
        )
        # Orientation audit: planned TCP RPY (deg, XYZ order) against
        # the straight Cartesian slerp at the same fractions. A joint
        # -space plan can land perfectly while twisting through a
        # different route mid-path; that divergence shows up here.
        n_wp = tcp_quat.shape[0]
        print(
            f"[INFO] plan ori path (planned-rpy vs slerp-rpy deg, N={n_wp}):",
            flush=True,
        )
        q0, q1 = tcp_quat[0], tcp_quat[-1]
        for f in (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0):
            qp = tcp_quat[min(int(round(f * (n_wp - 1))), n_wp - 1)]
            qs = math_utils.quat_slerp(q0, q1, float(f)).reshape(1, 4)
            rp = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(qp.unsqueeze(0)), dim=-1))[0]
            rs = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(qs), dim=-1))[0]
            print(
                f"[INFO]   f={f:.2f} plan={[round(float(v), 1) for v in rp.tolist()]} "
                f"slerp={[round(float(v), 1) for v in rs.tolist()]}",
                flush=True,
            )

    def _plan_from_reference(self, env_ids: torch.Tensor, robot_cfg: dict) -> None:
        """Retarget a dense Cartesian reference into joint waypoints.

        Instead of a free start-to-goal trajopt (which may shortcut the
        authored arc), solve IK along each reference frame with cuRobo's
        MotionRetargeter so execution follows the given trajectory.
        Falls back to direct servo on any failure.
        """
        from curobo.motion_retargeter import (
            MotionRetargeter,
            MotionRetargeterCfg,
            SequenceGoalToolPose,
        )
        from curobo.types import DeviceCfg, JointState, ToolPoseCriteria

        ref_pos = torch.tensor(self.cfg.reference_pos_env, device=self._device, dtype=torch.float32)
        ref_quat = torch.tensor(self.cfg.reference_quat_xyzw, device=self._device, dtype=torch.float32)
        num_waypoints = ref_pos.shape[0]
        tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
        cache_key = (
            "retargeter",
            self.cfg.robot_curobo_yaml,
            self.cfg.robot_urdf,
            self.cfg.num_ik_seeds,
        )

        def _make_retargeter() -> MotionRetargeter:
            with torch.inference_mode(False):
                return MotionRetargeter(
                    MotionRetargeterCfg.create(
                        robot=robot_cfg,
                        tool_pose_criteria={
                            tool_frame: ToolPoseCriteria.track_position_and_orientation(
                                xyz=[1.0, 1.0, 1.0],
                                rpy=[1.0, 1.0, 1.0],
                                non_terminal_scale=1.0,
                            )
                        },
                        num_envs=1,
                        use_mpc=False,
                        self_collision_check=False,
                        scene_model=None,
                        load_collision_spheres=False,
                        optimization_dt=self._dt,
                        num_seeds_global=self.cfg.num_ik_seeds,
                        num_seeds_local=1,
                        position_tolerance=0.002,
                        orientation_tolerance=0.02,
                        device_cfg=DeviceCfg(),
                    )
                )

        retargeter, build_s = self._command_term.get_curobo_solver(cache_key, _make_retargeter)
        solve_start = time.perf_counter()
        try:
            if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                raise ValueError("CuroboPlannedGoToFrameCfg requires pose_command.body_offset")
            # Single shared plan like the trajopt path: the reference is one
            # trajectory, executed synchronously by all envs.
            rep = env_ids[0:1]
            ref_pos_w = self._command_term._env.scene.env_origins[rep] + ref_pos
            tcp_pos_b, tcp_quat_b = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w.torch[rep].expand(num_waypoints, -1),
                self._asset.data.root_quat_w.torch[rep].expand(num_waypoints, -1),
                ref_pos_w,
                ref_quat.expand(num_waypoints, -1),
            )
            flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                tcp_pos_b, tcp_quat_b, rep
            )
            joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
            current = JointState.from_position(
                self._asset.data.joint_pos.torch[rep][:, joint_ids],
                joint_names=joint_names,
            )
            curobo_flange = retargeter.kinematics.compute_kinematics(current).tool_poses.get_link_pose(tool_frame)
            isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w.torch[rep],
                self._asset.data.root_quat_w.torch[rep],
                self._asset.data.body_pos_w.torch[rep, self._command_term._body_idx],
                self._asset.data.body_quat_w.torch[rep, self._command_term._body_idx],
            )
            # cuRobo quaternions are (w, x, y, z); math_utils is (x, y, z, w).
            flange_quat_inv_c = math_utils.quat_inv(_wxyz_to_xyzw(curobo_flange.quaternion))
            flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
            curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
                isaac_flange_pos_b,
                isaac_flange_quat_b,
                flange_pos_inv_c,
                flange_quat_inv_c,
            )
            flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                curobo_base_pos_b.expand(num_waypoints, -1),
                curobo_base_quat_b.expand(num_waypoints, -1),
                flange_pos_b,
                flange_quat_b,
            )
            with torch.inference_mode(False), torch.enable_grad():
                current = JointState.from_position(current.position.clone(), joint_names=joint_names)
                arc_targets = SequenceGoalToolPose(
                    tool_frames=[tool_frame],
                    position=flange_pos_c[:, None, None, None, :].clone(),
                    quaternion=_xyzw_to_wxyz(flange_quat_c[:, None, None, None, :].clone()),
                )
                if hasattr(retargeter, "_set_initial_joint_state"):
                    result = retargeter.solve_sequence(
                        arc_targets,
                        initial_joint_state=current,
                    )
                    joint_waypoints = result.joint_state.reorder(joint_names).position
                else:
                    # Compatibility with installed cuRobo releases that
                    # predate the initial_joint_state sequence API.
                    retargeter.reset()
                    retargeter._prev_solution = current.position.clone()
                    retargeter._prev_velocity = current.velocity.clone()
                    frame_solutions = [
                        retargeter.solve_frame(arc_targets.get_frame(index))
                        .joint_state.reorder(joint_names)
                        .position
                        for index in range(arc_targets.num_frames)
                    ]
                    joint_waypoints = torch.stack(frame_solutions, dim=1)
            max_joint_step = torch.max(torch.abs(joint_waypoints[:, 1:] - joint_waypoints[:, :-1]))
            if max_joint_step > 0.15:
                print(
                    f"[WARN] cuRobo reference retarget has a joint discontinuity "
                    f"(max step={float(max_joint_step.item()):.3f} rad); falling back to direct servo",
                    flush=True,
                )
                self._fallback = True
                return
            solve_s = time.perf_counter() - solve_start
            self._publish_plan(
                tcp_pos_b, tcp_quat_b, joint_waypoints, origin="retargeted", build_s=build_s, solve_s=solve_s
            )
        except Exception as err:
            try:
                seg = self._command_term._command_handlers.index(self)
            except ValueError:
                seg = "?"
            print(f"[WARN] cuRobo reference retarget failed on seg {seg} ({err}); falling back", flush=True)
            self._fallback = True

    def _validate_arc_radius(self, tcp_pos_b: torch.Tensor, wander_tol_m: float = 0.01) -> None:
        """Check the planned TCP path against its own endpoint chord.

        A plan between two bead targets should not wander beyond what the
        straight chord between its endpoints implies: compare the range of
        ``|TCP - pivot|`` along the plan vs along the chord. Warns when the
        plan bows out (bad arc / wrong pivot), not when the beads themselves
        imply varying radius. Display only; never fails the plan.
        """
        term = self._command_term
        valve = term._valve_asset
        if valve is None or tcp_pos_b.shape[0] < 2:
            return
        if not hasattr(self, "_valve_pivot_idx"):
            # Anchor = joint anchor (a fixed point ON the rotation axis).
            # Per Ball_Valve.urdf, RevoluteJoint's origin is identity in its
            # parent valvula_esfera, so that body's origin IS the anchor.
            self._valve_pivot_idx = None
            self._valve_pivot_name = "valve-root"
            for candidate in ("valvula_esfera", "alavanca_pivot"):
                try:
                    ids, _ = valve.find_bodies(candidate)
                except Exception:
                    ids = []
                if ids:
                    self._valve_pivot_idx = int(ids[0])
                    self._valve_pivot_name = candidate
                    break
        if self._valve_pivot_idx is None:
            pivot_w = valve.data.root_pos_w.torch[0]
        else:
            pivot_w = valve.data.body_pos_w.torch[0, self._valve_pivot_idx]
        pivot_name = self._valve_pivot_name
        n = tcp_pos_b.shape[0]
        root_pos = self._asset.data.root_pos_w.torch[0:1].expand(n, -1)
        root_quat = self._asset.data.root_quat_w.torch[0:1].expand(n, -1)
        identity = torch.zeros(n, 4, device=self._device)
        identity[:, 3] = 1.0
        tcp_w, _ = math_utils.combine_frame_transforms(root_pos, root_quat, tcp_pos_b, identity)

        def _range(t: torch.Tensor) -> float:
            d = torch.linalg.vector_norm(t - pivot_w.unsqueeze(0), dim=-1)
            return float(d.max().item() - d.min().item()), float(d.mean().item())

        plan_range, plan_mean = _range(tcp_w)
        f = torch.linspace(0.0, 1.0, n, device=self._device).unsqueeze(-1)
        chord = tcp_w[0:1] * (1.0 - f) + tcp_w[-1:] * f
        chord_range, _ = _range(chord)
        plan_mm, chord_mm = plan_range * 1000.0, chord_range * 1000.0
        try:
            seg = self._command_term._command_handlers.index(self)
        except ValueError:
            seg = "?"
        print(
            f"[INFO] plan seg {seg} arc radius vs {pivot_name} "
            f"@ {[round(float(v), 4) for v in pivot_w.detach().cpu().tolist()]}: "
            f"plan range={plan_mm:.1f}mm chord range={chord_mm:.1f}mm mean={plan_mean:.4f}m (N={n})",
            flush=True,
        )
        if plan_mm > chord_mm + wander_tol_m * 1000.0:
            print(
                f"[WARN] plan seg {seg} wanders {plan_mm - chord_mm:.1f}mm beyond its "
                f"endpoint chord: path not circular about the valve pivot",
                flush=True,
            )

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        if self._fallback:
            return super().update(env_mask)
        env_ids = torch.where(env_mask)[0]
        # CommandManager resets every sequence handler at episode reset. Plan
        # here instead, once this term actually becomes active after the arm
        # has reached the lower lever pose and closed the gripper.
        if self._waypoint_pos_b is None:
            self._plan(env_ids)
            if self._fallback:
                return super().update(env_mask)
        # CommandManager advances a completed handler and still invokes its
        # update once in that same tick.  Use the final waypoint for that
        # harmless trailing update while retaining the one-past-end completion
        # sentinel in ``_waypoint_index``.
        last = self._waypoint_pos_b.shape[1] - 1
        index = torch.clamp(self._waypoint_index[env_ids], max=last)
        pos = self._waypoint_pos_b[env_ids, index]
        quat = self._waypoint_quat_b[env_ids, index]
        self._joint_target[env_ids] = self._joint_waypoints[env_ids, index]
        # Keep one past the final waypoint as a completion sentinel.  Clamping
        # at ``last`` would mark the handler done before ever commanding the
        # final planned joint target.
        self._waypoint_index[env_ids] = index + 1
        return self._pack_command(self.cfg.gripper_open, pos, quat)

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self._fallback:
            return super().is_done(env_ids)
        if self._waypoint_pos_b is None:
            return torch.zeros(len(env_ids), device=self._device, dtype=torch.bool)
        exhausted = self._waypoint_index[env_ids] >= self._waypoint_pos_b.shape[1]
        if not torch.any(exhausted):
            return exhausted
        # Waypoint exhaustion alone is not convergence: the final waypoint is
        # first commanded the same tick the index exhausts, so advancing here
        # would chain the next plan from lagged joints. update() keeps
        # publishing the final waypoint/joints, so waiting for the parent
        # distance/orientation check lets the arm settle onto the plan end.
        # A plan that ends out of tolerance trips the stall watchdog instead.
        return exhausted & super().is_done(env_ids)


class _GripperHandler(_BaseCmdHandler):
    """Handles the GripperCommand."""

    def __init__(self, cfg: GripperCommand, command_term: SequentialPoseCommand):
        super().__init__(cfg, command_term)
        self.cfg: GripperCommand
        if self.cfg.duration_s <= 0.0:
            raise ValueError("GripperCommand.duration_s must be positive")
        self._elapsed_s = torch.zeros(self._num_envs, device=self._device)

    def reset(self, env_ids: torch.Tensor):
        self._elapsed_s[env_ids] = 0.0

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        self._elapsed_s[env_mask] += self._dt
        # Hold the last pose while changing only the gripper command.
        last_command = self._command_term._command[env_mask].clone()
        last_command[:, 0] = 1.0 if self.cfg.open_gripper else -1.0
        return last_command

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._elapsed_s[env_ids] >= self.cfg.duration_s

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return (
            self._command_term._command[env_ids, 1:4],
            self._command_term._command[env_ids, 4:8],
        )


class _RotateFrameHandler(_BaseCmdHandler):
    """Handles the RotateFrame command."""

    def __init__(
        self,
        cfg: RotateFrameCfg,
        command_term: SequentialPoseCommand,
    ):
        super().__init__(cfg, command_term)
        self.cfg: RotateFrameCfg
        if self.cfg.angular_velocity <= 0.0:
            raise ValueError("RotateFrameCfg.angular_velocity must be positive")
        if self.cfg.angle_threshold_deg < 0.0:
            raise ValueError("RotateFrameCfg.angle_threshold_deg must be non-negative")

        self._frame: FrameTransformer = command_term._env.scene[cfg.frame_name]
        self._frame_idx = self._frame.data.target_frame_names.index(cfg.target_frame_name)
        self.initial_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.axis_pos_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.axis_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self._progress_abs = torch.zeros(self._num_envs, device=self._device)
        self.angle_rad_tensor = torch.deg2rad(
            torch.tensor(self.cfg.angle_deg, device=self._device, dtype=torch.float32)
        ).repeat(self._num_envs)
        self.final_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.rot_axis_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.radius_vec = torch.zeros(self._num_envs, 3, device=self._device)
        self._angle_threshold_rad = math.radians(self.cfg.angle_threshold_deg)

    def reset(self, env_ids: torch.Tensor):
        self._progress_abs[env_ids] = 0.0

        if self._command_term._valve_asset is not None and self.cfg.use_valve_angle:
            self._command_term.recompute_valve_rotate_angle(env_ids)
            self.angle_rad_tensor[env_ids] = self._command_term.valve_rotate_angle_rad[env_ids]
        else:
            self.angle_rad_tensor[env_ids] = math.radians(self.cfg.angle_deg)

        ee_pos_b, self.initial_quat_b[env_ids] = self._command_term._get_ee_in_base_frame(env_ids)

        self.axis_pos_b[env_ids], axis_quat_b = self._get_rotation_axis_pose_b(env_ids)

        if self._command_term.cfg.debug_vis:
            print(
                "Initial Pose in relation to axis: ",
                ee_pos_b - self.axis_pos_b[env_ids],
            )
            print("Initial pose: ", ee_pos_b)
            print("Axis position: ", self.axis_pos_b[env_ids])
            print("Axis quat: ", axis_quat_b)

        # The rotation vector is explicit in the axis frame. Keeping it in the
        # command config prevents a grasp-frame rotation from silently changing
        # the mechanical joint axis.
        axis_in_frame = torch.tensor(self.cfg.axis, device=self._device, dtype=torch.float32).repeat(len(env_ids), 1)
        if torch.any(torch.linalg.vector_norm(axis_in_frame, dim=-1) < 1.0e-6):
            raise ValueError("RotateFrameCfg.axis must be non-zero")

        # Rotation axis expressed in the robot base frame.
        rot_axis_b = math_utils.quat_apply(axis_quat_b, axis_in_frame)
        self.rot_axis_b[env_ids] = rot_axis_b / torch.linalg.vector_norm(rot_axis_b, dim=-1, keepdim=True)
        if self._command_term.cfg.debug_vis:
            print("Rotation axis: ", self.rot_axis_b[env_ids])

        # vector from rotation center to initial ee pos, which describes
        # our expected motion
        radius_vec = ee_pos_b - self.axis_pos_b[env_ids]
        self.radius_vec[env_ids] = self._get_ortogonal_vector(radius_vec, self.rot_axis_b[env_ids])
        if self._command_term.cfg.debug_vis:
            print("Radius vector: ", self.radius_vec[env_ids])

        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]
        total_rotation = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        self.final_quat_b[env_ids] = math_utils.quat_mul(total_rotation, self.initial_quat_b[env_ids])

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]

        v_rot = self._rodrigues_rotate(self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle)
        final_pose_b = self.axis_pos_b[env_ids] + v_rot

        total_rotation = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        final_quat = math_utils.quat_mul(total_rotation, self.initial_quat_b[env_ids])

        return final_pose_b, final_quat

    def _rodrigues_rotate(self, v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return (
            v * torch.cos(theta)[:, None]
            + torch.cross(axis, v, dim=-1) * torch.sin(theta)[:, None]
            + axis * torch.sum(axis * v, dim=-1)[:, None] * (1 - torch.cos(theta))[:, None]
        )

    def _get_ortogonal_vector(self, vector: torch.Tensor, ortogonal_to: torch.Tensor) -> torch.Tensor:
        """
        Given a vector V and K, we can decompose on V = V || k  + V |_ k

        This function returns the V |_ k
        """
        return vector - torch.sum(vector * ortogonal_to, dim=-1, keepdim=True) * ortogonal_to

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        abs_angle = torch.abs(self.angle_rad_tensor[env_ids])
        self._progress_abs[env_ids] = torch.clamp(
            self._progress_abs[env_ids] + self.cfg.angular_velocity * self._dt,
            max=abs_angle,
        )
        angle = torch.copysign(self._progress_abs[env_ids], self.angle_rad_tensor[env_ids])

        # -- position
        # rotate radius vector using Rodrigues' rotation formula
        v_rot = self._rodrigues_rotate(self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle)
        target_pos_b = self.axis_pos_b[env_ids] + v_rot

        # Same signed angle-axis as the position orbit. Slerping to the
        # endpoint via quat_box_minus is ambiguous at ±180 deg and can spin
        # the gripper the opposite way from the TCP, so the wrist offset
        # walks through the hub instead of riding the grasp circle.
        delta_q = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        interp_quat_b = math_utils.quat_mul(delta_q, self.initial_quat_b[env_ids])

        return self._pack_command(self.cfg.gripper_open, target_pos_b, interp_quat_b)

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        remaining = torch.abs(self.angle_rad_tensor[env_ids]) - self._progress_abs[env_ids]
        return remaining <= self._angle_threshold_rad

    def _get_rotation_axis_pose_b(self, env_ids: torch.Tensor):
        # Target pose in base frame
        target_pos_w = self._frame.data.target_pos_w.torch[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w.torch[env_ids, self._frame_idx]

        target_pos_w, target_quat_w = self._apply_target_offset(target_pos_w, target_quat_w)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch[env_ids],
            self._asset.data.root_quat_w.torch[env_ids],
            target_pos_w,
            target_quat_w,
        )
        if self.cfg.axis_position_override_b is not None:
            for axis, value in enumerate(self.cfg.axis_position_override_b):
                if value is not None:
                    target_pos_b[:, axis] = float(value)

        return target_pos_b, target_quat_b


class _ScrewFrameHandler(_RotateFrameHandler):
    """Rotate around a frame while translating along its rotation axis."""

    def __init__(
        self,
        cfg: "ScrewFrameCfg",
        command_term: SequentialPoseCommand,
    ):
        super().__init__(cfg, command_term)
        self.cfg: ScrewFrameCfg
        self._joint_start = torch.zeros(self._num_envs, device=self._device)

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        command_term = self._command_term
        if command_term._screw_asset is not None and command_term._screw_revolute_idx is not None:
            self._joint_start[env_ids] = command_term._screw_asset.data.joint_pos.torch[
                env_ids, command_term._screw_revolute_idx
            ]

    def _axial_offset(self, env_ids: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        total_angle = torch.abs(self.angle_rad_tensor[env_ids]).clamp_min(1.0e-8)
        fraction = torch.abs(angle) / total_angle
        return self.rot_axis_b[env_ids] * (fraction * float(self.cfg.axial_distance))[:, None]

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        final_pos_b, final_quat_b = super().get_target_in_base_frame(env_ids)
        return (
            final_pos_b + self.rot_axis_b[env_ids] * float(self.cfg.axial_distance),
            final_quat_b,
        )

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        abs_angle = torch.abs(self.angle_rad_tensor[env_ids])
        self._progress_abs[env_ids] = torch.clamp(
            self._progress_abs[env_ids] + self.cfg.angular_velocity * self._dt,
            max=abs_angle,
        )
        angle = torch.copysign(self._progress_abs[env_ids], self.angle_rad_tensor[env_ids])

        coupling = self._command_term.cfg.screw_coupling
        if coupling is not None and coupling.command_joint_angle_scale != 0.0:
            self._command_term._screw_revolute_target[env_ids] = self._joint_start[env_ids] + angle * float(
                coupling.command_joint_angle_scale
            )

        v_rot = self._rodrigues_rotate(self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle)
        target_pos_b = self.axis_pos_b[env_ids] + v_rot + self._axial_offset(env_ids, angle)
        delta_q = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        target_quat_b = math_utils.quat_mul(delta_q, self.initial_quat_b[env_ids])
        return self._pack_command(self.cfg.gripper_open, target_pos_b, target_quat_b)


class _CuroboPlannedRotateFrameHandler(_CuroboPlannedGoToFrameHandler, _RotateFrameHandler):
    """Solve the complete valve arc as ordered cuRobo IK joint waypoints."""

    def __init__(self, cfg: "CuroboPlannedRotateFrameCfg", command_term: SequentialPoseCommand):
        # Initialize the mechanical rotation geometry, then the cuRobo state.
        _RotateFrameHandler.__init__(self, cfg, command_term)
        self.cfg: CuroboPlannedRotateFrameCfg
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self._fallback = False

    def reset(self, env_ids: torch.Tensor):
        _RotateFrameHandler.reset(self, env_ids)
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index[env_ids] = 0
        self._fallback = False
        if len(env_ids) != self._num_envs:
            raise RuntimeError("CuroboPlannedRotateFrameCfg currently requires synchronized environments.")

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return _RotateFrameHandler.get_target_in_base_frame(self, env_ids)

    def _plan(self, env_ids: torch.Tensor) -> None:
        """Retarget the complete Cartesian arc from the current grasp state."""
        if len(env_ids) != 1:
            raise RuntimeError("CuroboPlannedRotateFrameCfg currently supports one environment.")

        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        with curobo_compatible_warp():
            from curobo.motion_retargeter import (
                MotionRetargeter,
                MotionRetargeterCfg,
                SequenceGoalToolPose,
            )
            from curobo.types import DeviceCfg, JointState, ToolPoseCriteria

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
            cache_key = (
                "retargeter",
                self.cfg.robot_curobo_yaml,
                self.cfg.robot_urdf,
                self.cfg.num_ik_seeds,
            )

            def _make_retargeter() -> MotionRetargeter:
                with torch.inference_mode(False):
                    return MotionRetargeter(
                        MotionRetargeterCfg.create(
                            robot=robot_cfg,
                            tool_pose_criteria={
                                tool_frame: ToolPoseCriteria.track_position_and_orientation(
                                    xyz=[1.0, 1.0, 1.0],
                                    rpy=[1.0, 1.0, 1.0],
                                    non_terminal_scale=1.0,
                                )
                            },
                            num_envs=1,
                            use_mpc=False,
                            self_collision_check=False,
                            scene_model=None,
                            load_collision_spheres=False,
                            optimization_dt=self._dt,
                            num_seeds_global=self.cfg.num_ik_seeds,
                            num_seeds_local=1,
                            position_tolerance=0.002,
                            orientation_tolerance=0.02,
                            device_cfg=DeviceCfg(),
                        )
                    )

            retargeter, _ = self._command_term.get_curobo_solver(cache_key, _make_retargeter)

            if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                raise ValueError("CuroboPlannedRotateFrameCfg requires pose_command.body_offset")
            try:
                final_angle = self.angle_rad_tensor[env_ids][0]
                step_angle = self.cfg.angular_velocity * self._dt
                num_steps = max(1, int(math.ceil(abs(float(final_angle.item())) / step_angle)))
                num_waypoints = num_steps + 1
                angles = torch.linspace(
                    0.0,
                    float(final_angle.item()),
                    num_waypoints,
                    device=self._device,
                    dtype=torch.float32,
                )
                axis_b = self.rot_axis_b[env_ids].expand(num_waypoints, -1)
                radius_b = self.radius_vec[env_ids].expand(num_waypoints, -1)
                axis_pos_b = self.axis_pos_b[env_ids].expand(num_waypoints, -1)
                tcp_pos_b = axis_pos_b + self._rodrigues_rotate(radius_b, axis_b, angles)
                delta_quat_b = math_utils.quat_from_angle_axis(angles, axis_b)
                tcp_quat_b = math_utils.quat_mul(
                    delta_quat_b,
                    self.initial_quat_b[env_ids].expand(num_waypoints, -1),
                )

                flange_pos_b, flange_quat_b = self._command_term._tcp_pose_to_body_pose(
                    tcp_pos_b, tcp_quat_b, env_ids
                )

                joint_ids, joint_names = self._asset.find_joints(self.cfg.robot_joint_names, preserve_order=True)
                current = JointState.from_position(
                    self._asset.data.joint_pos.torch[env_ids][:, joint_ids],
                    joint_names=joint_names,
                )

                curobo_flange = retargeter.kinematics.compute_kinematics(current).tool_poses.get_link_pose(tool_frame)
                isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                    self._asset.data.root_pos_w.torch[env_ids],
                    self._asset.data.root_quat_w.torch[env_ids],
                    self._asset.data.body_pos_w.torch[env_ids, self._command_term._body_idx],
                    self._asset.data.body_quat_w.torch[env_ids, self._command_term._body_idx],
                )
                # cuRobo quaternions are (w, x, y, z); math_utils is (x, y, z, w).
                flange_quat_inv_c = math_utils.quat_inv(_wxyz_to_xyzw(curobo_flange.quaternion))
                flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
                curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
                    isaac_flange_pos_b,
                    isaac_flange_quat_b,
                    flange_pos_inv_c,
                    flange_quat_inv_c,
                )
                flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
                    curobo_base_pos_b.expand(num_waypoints, -1),
                    curobo_base_quat_b.expand(num_waypoints, -1),
                    flange_pos_b,
                    flange_quat_b,
                )

                with torch.inference_mode(False), torch.enable_grad():
                    current = JointState.from_position(current.position.clone(), joint_names=joint_names)
                    arc_targets = SequenceGoalToolPose(
                        tool_frames=[tool_frame],
                        position=flange_pos_c[:, None, None, None, :].clone(),
                        quaternion=_xyzw_to_wxyz(flange_quat_c[:, None, None, None, :].clone()),
                    )
                    if hasattr(retargeter, "_set_initial_joint_state"):
                        result = retargeter.solve_sequence(
                            arc_targets,
                            initial_joint_state=current,
                        )
                        joint_waypoints = result.joint_state.reorder(joint_names).position
                    else:
                        # Compatibility with installed cuRobo releases that
                        # predate the initial_joint_state sequence API. This is
                        # equivalent to solve_sequence(), but preserves the
                        # measured grasp branch before calling solve_frame().
                        retargeter.reset()
                        retargeter._prev_solution = current.position.clone()
                        retargeter._prev_velocity = current.velocity.clone()
                        frame_solutions = [
                            retargeter.solve_frame(arc_targets.get_frame(index))
                            .joint_state.reorder(joint_names)
                            .position
                            for index in range(arc_targets.num_frames)
                        ]
                        joint_waypoints = torch.stack(frame_solutions, dim=1)

                max_joint_step = torch.max(torch.abs(joint_waypoints[:, 1:] - joint_waypoints[:, :-1]))
                if max_joint_step > self.cfg.max_joint_step:
                    raise RuntimeError(
                        "cuRobo retargeted arc contains a joint discontinuity: "
                        f"max joint step={float(max_joint_step.item()):.3f} rad"
                    )

                self._waypoint_pos_b = tcp_pos_b.unsqueeze(0)
                self._waypoint_quat_b = tcp_quat_b.unsqueeze(0)
                self._joint_waypoints = joint_waypoints
                self._joint_target = torch.zeros(
                    self._num_envs,
                    len(joint_names),
                    device=self._device,
                    dtype=self._joint_waypoints.dtype,
                )
                print(f"[INFO] cuRobo retargeted {num_waypoints} ordered waypoints for the valve arc.")
            except Exception as err:
                # A throwing solver must not kill the episode: degrade to
                # direct servo like a failed plan.
                print(
                    f"[WARN] cuRobo arc retarget threw ({type(err).__name__}: {err}); falling back",
                    flush=True,
                )
                self._fallback = True
                return


@configclass
class ScrewJointCouplingCfg:
    """Environment-side equivalent of the lamp USD's screw ActionGraph."""

    asset_name: str = MISSING  # type: ignore
    revolute_joint_name: str = "RevoluteJoint"
    prismatic_joint_name: str = "PrismaticJoint"
    pitch_m_per_revolution: float = 0.006
    command_joint_angle_scale: float = 0.0
    lower_limit: float = 0.0
    upper_limit: float = 0.024
    viscous_friction: float = 2.0
    coulomb_friction: float = 1.0
    stiction: float = 3.0
    velocity_epsilon: float = 0.01
    end_stop_base_damping: float = 10.0
    end_stop_scale: float = 0.02
    end_stop_power: float = 2.5
    end_stop_activation_distance: float = 0.005


@configclass
class SequentialPoseCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "fr3_hand" (or similar) frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation ``(x, y, z, w)`` w.r.t. the parent frame. Defaults to identity."""

    class_type: type = SequentialPoseCommand

    asset_name: str = MISSING  # type: ignore
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING  # type: ignore
    """Name of the end-effector body used for pose commands."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    commands: Sequence[BaseCmd] = MISSING  # type: ignore
    """The sequence of commands to execute."""

    screw_coupling: ScrewJointCouplingCfg | None = None
    """Optional measured revolute-to-prismatic screw coupling."""

    valve_asset_name: str | None = None
    """Valve articulation to use for per-episode open/close tasks."""
    valve_joint_name: str = "RevoluteJoint"
    """Valve joint controlled by the task."""
    open_task_prob: float = 0.5
    """Probability of selecting the open endpoint when both directions are feasible."""
    valve_joint_closed: float = 0.0
    """Joint position representing the closed endpoint [rad]."""
    valve_joint_open: float = -math.pi / 2
    """Joint position representing the open endpoint [rad]."""
    valve_min_delta_rad: float = math.radians(20.0)
    """Minimum required distance between the sampled start and desired endpoint."""
    valve_ee_joint_angle_scale: float = 1.0
    """Scale from remaining valve error to the commanded EE arc angle."""

    output_joint_positions: bool = False
    """If True, ``command`` is cuRobo joint waypoints plus the gripper bit.

    Handlers still track a TCP pose internally. Between plans the last
    joints are held. Matches a ``JointPositionAction`` arm term plus gripper.
    """
    ik_joint_names: Sequence[str] | None = None
    """Arm joints for ``output_joint_positions``. Defaults to the cuRobo command's names."""

    debug_vis: bool = False

    log_transitions: bool = False
    """Log segment start/finish with per-segment durations."""

    stall_timeout_s: float = 0.0
    """Warn + dump diagnostics when a segment exceeds this duration [s]. 0 disables."""

    path_debug_vis: bool = False
    """Show the active GoTo/Rotate/Screw path or cuRobo plan and next goal. Needs debug_vis."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/pose_goal"
    )

    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(  # type: ignore
        prim_path="/Visuals/Command/current_pose"
    )

    """The configuration for the current pose visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)  # type: ignore


@configclass
class BaseCmd:
    """Base configuration for frame pose command generators."""

    class_type: Type[_BaseCmdHandler] = MISSING  # type: ignore


@configclass
class GoToFrameCfg(BaseCmd):
    class_type = _GoToFrameHandler

    frame_name: str = MISSING  # type: ignore
    """Name of the frame used for pose commands."""
    target_frame_name: str = MISSING  # type: ignore
    """Index of the frame used for pose commands."""
    target_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Goal translation in the selected target frame [m]; ignored for fixed env goals."""
    target_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Goal rotation (xyzw) in the selected target frame; ignored for fixed env goals."""
    target_position_env: tuple[float, float, float] | None = None
    """Optional fixed position relative to the environment origin.

    When set, bypass the frame sensor and retain the orientation at reset.
    Set both frame names to empty strings for this mode.
    """
    target_orientation_env: tuple[float, float, float, float] | None = None
    """Fixed target quaternion (xyzw), used with target_position_env.

    None retains the orientation at reset. Environment axes match world axes.
    """
    gripper_open: bool = False
    """Status of the gripper during the command."""
    velocity: float = 0.2
    """Linear speed of the commanded pose toward the target frame [m/s]."""
    angular_velocity: float = 1.0
    """Angular speed of the commanded orientation toward the target [rad/s]."""
    distance_threshold: float = 0.05
    """End-effector distance to the target frame at which the command is done [m]."""
    orientation_threshold_deg: float = 10.0
    """End-effector orientation error to the target at which the command is done [deg]."""
    valve_done_threshold_rad: float | None = None
    """Also complete once the valve is within this of its desired angle [rad].

    For contact-rich segments (e.g. the valve arc) the TCP can hold centimeters
    off-target while the task is physically done. None disables.
    """
    canonicalize_upward: bool = True
    """If True, choose the upright TCP +X half-turn once at the segment start.

    Keep that choice while following the moving reference so crossing a
    horizontal pose does not abruptly reverse the orientation target.

    That assumes Spot's TCP (+Z up, +X approach). Franka TCP is +Z approach,
    +X hand-top — leave this False there so the authored frame rotation is
    used as-is.
    """
    hold_current_orientation: bool = False
    """Translate to the target while retaining the orientation captured at reset."""
    position_override_b: tuple[float | None, float | None, float | None] | None = None
    """Optional per-axis target-position overrides in the robot base frame."""


@configclass
class CuroboPlannedGoToFrameCfg(GoToFrameCfg):
    """A cuRobo-planned variant of :class:`GoToFrameCfg`."""

    class_type = _CuroboPlannedGoToFrameHandler
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    num_trajopt_seeds: int = 1
    max_plan_attempts: int = 1
    interpolation_buffer_size: int = 128
    reference_pos_env: tuple[tuple[float, float, float], ...] | None = None
    """Dense env-frame TCP positions steering the plan (retargeted, not free)."""
    reference_quat_xyzw: tuple[tuple[float, float, float, float], ...] | None = None
    """Dense TCP orientations (x, y, z, w) matching :attr:`reference_pos_env`."""


@configclass
class GripperCommand(BaseCmd):
    class_type = _GripperHandler

    open_gripper: bool = True
    """Whether to open or close the gripper."""
    duration_s: float = 0.2
    """Hold time after changing the gripper command [s]."""


@configclass
class RotateFrameCfg(BaseCmd):
    class_type = _RotateFrameHandler

    frame_name: str = MISSING  # type: ignore
    """Name of the frame used to rotate around."""
    target_frame_name: str = "rotate_frame"
    """Name of the frame used to rotate around."""
    target_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Rotation-center translation in the selected reference frame [m]."""
    target_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Rotation-axis frame orientation (xyzw) relative to the selected reference."""
    axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    """Rotation axis expressed in ``target_frame_name`` coordinates."""
    angle_deg: float = -90.0
    """Angle in degrees to rotate around :attr:`axis`."""
    use_valve_angle: bool = True
    """Use the remaining task valve angle when available; False uses angle_deg."""
    angular_velocity: float = 0.3
    """Angular speed of the commanded arc [rad/s]."""
    gripper_open: bool = False
    """Status of the gripper during the command."""
    angle_threshold_deg: float = 5.0
    """Remaining commanded arc at which the command is done [deg]."""
    axis_position_override_b: tuple[float | None, float | None, float | None] | None = None
    """Optional per-axis rotation-center overrides in the robot base frame."""


@configclass
class ScrewFrameCfg(RotateFrameCfg):
    """Rotate around a frame while advancing along the configured axis."""

    class_type = _ScrewFrameHandler

    axial_distance: float = 0.0
    """Signed translation along :attr:`axis` over the full rotation [m]."""


@configclass
class CuroboPlannedRotateFrameCfg(RotateFrameCfg):
    """A cuRobo-retargeted arc variant of :class:`RotateFrameCfg`."""

    class_type = _CuroboPlannedRotateFrameHandler
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    interpolation_buffer_size: int = 128
    max_joint_step: float = 0.15
    """Maximum accepted change of any joint between arc waypoints [rad]."""
