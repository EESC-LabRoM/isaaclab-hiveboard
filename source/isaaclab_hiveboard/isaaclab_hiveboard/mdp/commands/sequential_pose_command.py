from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Sequence, Type

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
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


class SequentialPoseCommand(CommandTerm):
    """A command term that executes a sequence of commands for a robot's end-effector."""

    cfg: "SequentialPoseCommandCfg"

    def __init__(self, cfg: "SequentialPoseCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # obtain the robot asset
        # -- robot
        self._asset: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if not body_ids:
            raise ValueError(
                f"Body with name '{self.cfg.body_name}' not found in asset '{self.cfg.asset_name}'."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # -- command sequence
        self._current_command_idx = torch.zeros(
            self._env.num_envs, device=self._env.device, dtype=torch.long
        )
        self._command_handlers = [
            cmd.class_type(cmd, self) for cmd in self.cfg.commands
        ]

        # -- build default command, quat can't be 0s
        self._command = torch.zeros(
            (self._env.num_envs, 8), device=self._env.device, dtype=torch.float32
        )
        self._command[:, 0] = 1  # Close gripper
        self._command[:, 4] = 1.0  # (w,x,y,z) -> (1,0,0,0)

        # -- convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(
                self.cfg.body_offset.pos, device=self.device
            ).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(
                self.cfg.body_offset.rot, device=self.device
            ).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # -- optional valve task state
        self.valve_task_goal = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.valve_joint_start = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.valve_joint_des = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.valve_rotate_angle_rad = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._valve_asset: Articulation | None = None
        self._valve_joint_idx: int | None = None
        self._initialize_valve_task()

        if self.cfg.debug_vis:
            self._target_pos_b = torch.zeros(
                (self._env.num_envs, 3), device=self._env.device, dtype=torch.float32
            )
            self._target_quat_b = torch.zeros(
                (self._env.num_envs, 4), device=self._env.device, dtype=torch.float32
            )
            self._target_quat_b[:, 0] = 1.0  # (w,x,y,z) -> (1,0,0,0)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired command.

        The command is an 8-dimensional tensor:
        - 1 dimension for gripper status (1 for close, -1 for open)
        - 3 dimensions for the target end-effector position in the base frame
        - 4 dimensions for the target end-effector orientation (quat) in the base frame
        """
        return self._command

    def _resample_command(
        self, env_ids: Sequence[int] | slice | None | torch.Tensor = None
    ):
        """Resets the command sequence for the specified environments."""
        if isinstance(env_ids, slice) or env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._sample_valve_task(env_ids)
        self._current_command_idx[env_ids] = 0
        for handler in self._command_handlers:
            handler.reset(env_ids)
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

    def _initialize_valve_task(self) -> None:
        """Resolve and validate the optional valve articulation and task limits."""
        if self.cfg.valve_asset_name is None:
            return
        if self.cfg.valve_asset_name not in self._env.scene.keys():
            raise ValueError(
                f"Valve asset '{self.cfg.valve_asset_name}' not found in scene."
            )
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

        q_start = self._valve_asset.data.joint_pos[env_ids, self._valve_joint_idx]
        q_open = torch.full_like(q_start, self.cfg.valve_joint_open)
        q_closed = torch.full_like(q_start, self.cfg.valve_joint_closed)
        min_delta = float(self.cfg.valve_min_delta_rad)
        can_open = torch.abs(q_open - q_start) >= min_delta
        can_close = torch.abs(q_closed - q_start) >= min_delta
        if torch.any(~(can_open | can_close)):
            bad_q = q_start[~(can_open | can_close)].detach().cpu().tolist()
            raise RuntimeError(
                "Reset valve states leave no endpoint satisfying "
                f"valve_min_delta_rad={min_delta}: {bad_q}"
            )

        open_mask = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.open_task_prob
        )
        open_mask = torch.where(can_open & ~can_close, True, open_mask)
        open_mask = torch.where(can_close & ~can_open, False, open_mask)
        goal = torch.where(
            open_mask, torch.ones_like(q_start), -torch.ones_like(q_start)
        )
        q_des = torch.where(open_mask, q_open, q_closed)

        self.valve_task_goal[env_ids] = goal
        self.valve_joint_start[env_ids] = q_start
        self.valve_joint_des[env_ids] = q_des
        self.recompute_valve_rotate_angle(env_ids)

    def recompute_valve_rotate_angle(self, env_ids: torch.Tensor) -> None:
        """Update the EE arc angle from the valve's remaining joint error."""
        if self._valve_asset is None or self._valve_joint_idx is None:
            return
        q_current = self._valve_asset.data.joint_pos[env_ids, self._valve_joint_idx]
        self.valve_rotate_angle_rad[env_ids] = (
            self.valve_joint_des[env_ids] - q_current
        ) * float(self.cfg.valve_ee_joint_angle_scale)

    def _update_command(self):
        """Updates the command based on the current state of the command sequence."""
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
                self._current_command_idx[done_env_ids] = (
                    self._current_command_idx[done_env_ids] + 1
                )
                if i < (len(self._command_handlers) - 1):
                    self._command_handlers[i + 1].reset(done_env_ids)

            # update the command for the current handler
            self._command[env_mask] = handler.update(env_mask)

            if self.cfg.debug_vis:
                target_pos_b, target_quat_b = handler.get_target_in_base_frame(
                    torch.where(env_mask)[0]
                )
                self._target_pos_b[env_mask] = target_pos_b
                self._target_quat_b[env_mask] = target_quat_b

    def is_done(self) -> torch.Tensor:
        """Check if all commands were finished."""
        return self._current_command_idx == len(self._command_handlers)

    def get_curobo_joint_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return direct joint targets for environments executing a cuRobo term.

        The normal command remains a TCP pose so that the existing gripper and
        differential-IK action interface is unchanged.  During a planned
        segment, though, applying that pose through DLS can select a different
        redundant-joint branch than the one cuRobo validated.  The matching
        action term uses this method to apply the planner's exact joint
        waypoint instead.
        """
        targets = torch.zeros(
            self.num_envs, 0, device=self.device, dtype=self._command.dtype
        )
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

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
                # -- current
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )

            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self._asset.is_initialized:
            return

        asset_pos = self._asset.data.root_pos_w
        asset_quat = self._asset.data.root_quat_w

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

    def _get_ee_in_base_frame(self, env_ids: torch.Tensor | slice):
        """
        To convert the end-effector pose from world frame to base frame, we do:
        1. Get the end-effector pose in world frame   (Pwe)
        2. Subtract the base (root) pose from the end-effector pose (Pbe = Pwb^-1 * Pwe)
        3. Apply the fixed offset (if any) (Pbe' = Pbe * P_ee')
        """

        # End-effector pose in base frame
        ee_pos_w = self._asset.data.body_pos_w[env_ids, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[env_ids, self._body_idx]

        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            ee_pos_w,
            ee_quat_w,
        )

        if self._offset_pos is not None and self._offset_rot is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b,
                ee_quat_b,
                self._offset_pos[env_ids],
                self._offset_rot[env_ids],
            )
        return ee_pos_b, ee_quat_b

    def _get_ee_in_world_frame(self, env_ids: torch.Tensor | slice):
        """
        To convert the end-effector pose from world frame to world frame, we do:
        """
        # End-effector pose in world frame
        ee_pos_w = self._asset.data.body_pos_w[env_ids, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[env_ids, self._body_idx]

        if self._offset_pos is not None and self._offset_rot is not None:
            ee_pos_w, ee_quat_w = math_utils.combine_frame_transforms(
                ee_pos_w,  # T01
                ee_quat_w,  # R01
                self._offset_pos[env_ids],  # T12
                self._offset_rot[env_ids],  # R12
            )

        return ee_pos_w, ee_quat_w


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

    def _pack_command(
        self, gripper_open: bool, pos: torch.Tensor, quat: torch.Tensor
    ) -> torch.Tensor:
        command = torch.zeros(pos.shape[0], 8, device=self._device)
        command[:, 0] = 1.0 if gripper_open else -1.0
        command[:, 1:4] = pos
        command[:, 4:8] = quat
        return command

    def _step_pos_towards(
        self, current: torch.Tensor, target: torch.Tensor, velocity: float
    ) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``velocity * dt``."""
        delta = target - current
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(velocity * self._dt / dist.clamp(min=1.0e-8), max=1.0)
        return current + delta * scale

    def _step_quat_towards(
        self, current: torch.Tensor, target: torch.Tensor, angular_velocity: float
    ) -> torch.Tensor:
        """Advance ``current`` toward ``target`` by at most ``angular_velocity * dt``."""
        err = math_utils.quat_box_minus(target, current)
        angle = torch.linalg.vector_norm(err, dim=-1, keepdim=True)
        scale = torch.clamp(
            angular_velocity * self._dt / angle.clamp(min=1.0e-8), max=1.0
        )
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
            raise ValueError(
                "GoToFrameCfg.orientation_threshold_deg must be non-negative"
            )

        self._frame = command_term._env.scene[cfg.frame_name]
        self._frame_idx = self._frame.data.target_frame_names.index(
            cfg.target_frame_name
        )
        self.command_pos_b = torch.zeros(self._num_envs, 3, device=self._device)
        self.command_quat_b = torch.zeros(self._num_envs, 4, device=self._device)
        self.command_quat_b[:, 0] = 1.0
        self._ori_threshold_rad = math.radians(self.cfg.orientation_threshold_deg)

    def reset(self, env_ids: torch.Tensor):
        ee_pos_b, ee_quat_b = self._command_term._get_ee_in_base_frame(env_ids)
        self.command_pos_b[env_ids] = ee_pos_b
        self.command_quat_b[env_ids] = ee_quat_b

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
        return (pos_err <= self.cfg.distance_threshold) & (
            ori_err <= self._ori_threshold_rad
        )

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Target pose in base frame
        target_pos_w = self._frame.data.target_pos_w[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w[env_ids, self._frame_idx]

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            target_pos_w,
            target_quat_w,
        )
        if self.cfg.canonicalize_upward:
            target_quat_b = canonicalize_ee_orientation_upward(target_quat_b)

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

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index[env_ids] = 0
        if len(env_ids) != self._num_envs:
            raise RuntimeError("CuroboPlannedGoToFrameCfg currently requires synchronized environments.")

    def _plan(self, env_ids: torch.Tensor) -> None:
        """Build a bounded cuRobo plan and convert it to TCP pose waypoints."""
        from isaaclab_hiveboard.assets import ASSET_DIR
        from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
        from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

        with curobo_compatible_warp():
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

            robot_cfg = load_curobo_robot_cfg(
                self.cfg.robot_curobo_yaml or f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
                self.cfg.robot_urdf or f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            )
            # CommandManager runs in inference mode. Constructing cuRobo in
            # that mode makes its reusable goal buffers inference tensors,
            # which TrajOpt cannot subsequently update in place.
            with torch.inference_mode(False):
                planner = MotionPlanner(
                    MotionPlannerCfg.create(
                        robot=robot_cfg,
                        self_collision_check=False,
                        use_cuda_graph=False,
                        num_ik_seeds=self.cfg.num_ik_seeds,
                        num_trajopt_seeds=1,
                        interpolation_dt=self._dt,
                        interpolation_buffer_size=self.cfg.interpolation_buffer_size,
                        device_cfg=DeviceCfg(),
                    )
                )
            try:
                target_pos_b, target_quat_b = self.get_target_in_base_frame(env_ids)
                if self._command_term._offset_pos is None or self._command_term._offset_rot is None:
                    raise ValueError("CuroboPlannedGoToFrameCfg requires pose_command.body_offset")
                tcp_to_flange_quat = math_utils.quat_inv(
                    self._command_term._offset_rot[env_ids]
                )
                tcp_to_flange_pos = -math_utils.quat_apply(
                    tcp_to_flange_quat, self._command_term._offset_pos[env_ids]
                )
                flange_pos_b, flange_quat_b = math_utils.combine_frame_transforms(
                    target_pos_b,
                    target_quat_b,
                    tcp_to_flange_pos,
                    tcp_to_flange_quat,
                )
                joint_ids, joint_names = self._asset.find_joints(
                    self.cfg.robot_joint_names, preserve_order=True
                )
                current = JointState.from_position(
                    self._asset.data.joint_pos[env_ids][:, joint_ids], joint_names=joint_names
                )
                # The Isaac articulation root and the FR3 URDF root are not
                # the same frame.  Calibrate the constant transform from the
                # current shared joint configuration, then express the Isaac
                # target flange pose in cuRobo's URDF base frame.
                curobo_flange = planner.compute_kinematics(current).tool_poses.get_link_pose(
                    planner.tool_frames[0]
                )
                isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
                    self._asset.data.root_pos_w[env_ids],
                    self._asset.data.root_quat_w[env_ids],
                    self._asset.data.body_pos_w[env_ids, self._command_term._body_idx],
                    self._asset.data.body_quat_w[env_ids, self._command_term._body_idx],
                )
                flange_quat_inv_c = math_utils.quat_inv(curobo_flange.quaternion)
                flange_pos_inv_c = -math_utils.quat_apply(
                    flange_quat_inv_c, curobo_flange.position
                )
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
                    current = JointState.from_position(
                        current.position.clone(), joint_names=joint_names
                    )
                    goal = GoalToolPose.from_poses(
                        {
                            planner.tool_frames[0]: Pose(
                                position=flange_pos_c.clone(),
                                quaternion=flange_quat_c.clone(),
                            )
                        },
                        ordered_tool_frames=planner.tool_frames,
                        num_goalset=1,
                    )
                    result = planner.plan_pose(
                        goal, current, max_attempts=1, enable_graph_attempt=99
                    )
                if result is None or not bool(result.success.all().item()):
                    print(
                        "[WARN] cuRobo plan failed; "
                        f"start_q={current.position.detach().cpu().tolist()} "
                        f"isaac_flange={isaac_flange_pos_b.detach().cpu().tolist()} "
                        f"curobo_flange={curobo_flange.position.detach().cpu().tolist()} "
                        f"curobo_base={curobo_base_pos_b.detach().cpu().tolist()} "
                        f"flange_pos_c={flange_pos_c.detach().cpu().tolist()} "
                        f"flange_quat_c={flange_quat_c.detach().cpu().tolist()}"
                    )
                    raise RuntimeError("cuRobo could not plan the requested frame motion")
                trajectory = result.get_interpolated_plan().position
                if trajectory.ndim == 4:
                    trajectory = trajectory[:, 0]
                poses = planner.compute_kinematics(
                    JointState.from_position(trajectory[0], joint_names=joint_names)
                ).tool_poses.get_link_pose(planner.tool_frames[0])
                tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
                    poses.position,
                    poses.quaternion,
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
                self._waypoint_pos_b = tcp_pos.unsqueeze(0)
                self._waypoint_quat_b = tcp_quat.unsqueeze(0)
                self._joint_waypoints = trajectory
                self._joint_target = torch.zeros(
                    self._num_envs, trajectory.shape[-1], device=self._device, dtype=trajectory.dtype
                )
                print(f"[INFO] cuRobo planned {tcp_pos.shape[0]} TCP waypoints for {self.cfg.target_frame_name}.")
            finally:
                planner.destroy()

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        # CommandManager resets every sequence handler at episode reset. Plan
        # here instead, once this term actually becomes active after the arm
        # has reached the lower lever pose and closed the gripper.
        if self._waypoint_pos_b is None:
            self._plan(env_ids)
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
        if self._waypoint_pos_b is None:
            return torch.zeros(len(env_ids), device=self._device, dtype=torch.bool)
        return self._waypoint_index[env_ids] >= self._waypoint_pos_b.shape[1]


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
        last_command = self._command_term.command[env_mask].clone()
        last_command[:, 0] = 1.0 if self.cfg.open_gripper else -1.0
        return last_command

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self._elapsed_s[env_ids] >= self.cfg.duration_s

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return (
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
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
        self._frame_idx = self._frame.data.target_frame_names.index(
            cfg.target_frame_name
        )
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

        if self._command_term._valve_asset is not None:
            self._command_term.recompute_valve_rotate_angle(env_ids)
            self.angle_rad_tensor[env_ids] = self._command_term.valve_rotate_angle_rad[
                env_ids
            ]
        else:
            self.angle_rad_tensor[env_ids] = math.radians(self.cfg.angle_deg)

        ee_pos_b, self.initial_quat_b[env_ids] = (
            self._command_term._get_ee_in_base_frame(env_ids)
        )

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
        axis_in_frame = torch.tensor(
            self.cfg.axis, device=self._device, dtype=torch.float32
        ).repeat(len(env_ids), 1)
        if torch.any(torch.linalg.vector_norm(axis_in_frame, dim=-1) < 1.0e-6):
            raise ValueError("RotateFrameCfg.axis must be non-zero")

        # Rotation axis expressed in the robot base frame.
        rot_axis_b = math_utils.quat_apply(axis_quat_b, axis_in_frame)
        self.rot_axis_b[env_ids] = rot_axis_b / torch.linalg.vector_norm(
            rot_axis_b, dim=-1, keepdim=True
        )
        if self._command_term.cfg.debug_vis:
            print("Rotation axis: ", self.rot_axis_b[env_ids])

        # vector from rotation center to initial ee pos, which describes
        # our expected motion
        radius_vec = ee_pos_b - self.axis_pos_b[env_ids]
        self.radius_vec[env_ids] = self._get_ortogonal_vector(
            radius_vec, self.rot_axis_b[env_ids]
        )
        if self._command_term.cfg.debug_vis:
            print("Radius vector: ", self.radius_vec[env_ids])

        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]
        total_rotation = math_utils.quat_from_angle_axis(
            angle, self.rot_axis_b[env_ids]
        )
        self.final_quat_b[env_ids] = math_utils.quat_mul(
            total_rotation, self.initial_quat_b[env_ids]
        )

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        # Get final motion poses
        angle = self.angle_rad_tensor[env_ids]

        v_rot = self._rodrigues_rotate(
            self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle
        )
        final_pose_b = self.axis_pos_b[env_ids] + v_rot

        total_rotation = math_utils.quat_from_angle_axis(
            angle, self.rot_axis_b[env_ids]
        )
        final_quat = math_utils.quat_mul(total_rotation, self.initial_quat_b[env_ids])

        return final_pose_b, final_quat

    def _rodrigues_rotate(
        self, v: torch.Tensor, axis: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return (
            v * torch.cos(theta)[:, None]
            + torch.cross(axis, v, dim=-1) * torch.sin(theta)[:, None]
            + axis
            * torch.sum(axis * v, dim=-1)[:, None]
            * (1 - torch.cos(theta))[:, None]
        )

    def _get_ortogonal_vector(
        self, vector: torch.Tensor, ortogonal_to: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a vector V and K, we can decompose on V = V || k  + V |_ k

        This function returns the V |_ k
        """
        return (
            vector
            - torch.sum(vector * ortogonal_to, dim=-1, keepdim=True) * ortogonal_to
        )

    def update(self, env_mask: torch.Tensor) -> torch.Tensor:
        env_ids = torch.where(env_mask)[0]
        abs_angle = torch.abs(self.angle_rad_tensor[env_ids])
        self._progress_abs[env_ids] = torch.clamp(
            self._progress_abs[env_ids] + self.cfg.angular_velocity * self._dt,
            max=abs_angle,
        )
        angle = torch.copysign(
            self._progress_abs[env_ids], self.angle_rad_tensor[env_ids]
        )

        # -- position
        # rotate radius vector using Rodrigues' rotation formula
        v_rot = self._rodrigues_rotate(
            self.radius_vec[env_ids], self.rot_axis_b[env_ids], angle
        )
        target_pos_b = self.axis_pos_b[env_ids] + v_rot

        # Same signed angle-axis as the position orbit. Slerping to the
        # endpoint via quat_box_minus is ambiguous at ±180 deg and can spin
        # the gripper the opposite way from the TCP, so the wrist offset
        # walks through the hub instead of riding the grasp circle.
        delta_q = math_utils.quat_from_angle_axis(angle, self.rot_axis_b[env_ids])
        interp_quat_b = math_utils.quat_mul(delta_q, self.initial_quat_b[env_ids])

        return self._pack_command(self.cfg.gripper_open, target_pos_b, interp_quat_b)

    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        remaining = (
            torch.abs(self.angle_rad_tensor[env_ids]) - self._progress_abs[env_ids]
        )
        return remaining <= self._angle_threshold_rad

    def _get_rotation_axis_pose_b(self, env_ids: torch.Tensor):
        # Target pose in base frame
        target_pos_w = self._frame.data.target_pos_w[env_ids, self._frame_idx]
        target_quat_w = self._frame.data.target_quat_w[env_ids, self._frame_idx]

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            target_pos_w,
            target_quat_w,
        )

        return target_pos_b, target_quat_b


class _CuroboPlannedRotateFrameHandler(
    _CuroboPlannedGoToFrameHandler, _RotateFrameHandler
):
    """Execute a valve rotation endpoint through cuRobo joint waypoints."""

    def __init__(
        self, cfg: "CuroboPlannedRotateFrameCfg", command_term: SequentialPoseCommand
    ):
        # Initialize the mechanical rotation geometry, then the cuRobo state.
        _RotateFrameHandler.__init__(self, cfg, command_term)
        self.cfg: CuroboPlannedRotateFrameCfg
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index = torch.zeros(
            self._num_envs, dtype=torch.long, device=self._device
        )

    def reset(self, env_ids: torch.Tensor):
        _RotateFrameHandler.reset(self, env_ids)
        self._waypoint_pos_b = None
        self._waypoint_quat_b = None
        self._joint_waypoints = None
        self._joint_target = None
        self._waypoint_index[env_ids] = 0
        if len(env_ids) != self._num_envs:
            raise RuntimeError(
                "CuroboPlannedRotateFrameCfg currently requires synchronized environments."
            )

    def get_target_in_base_frame(self, env_ids: torch.Tensor):
        return _RotateFrameHandler.get_target_in_base_frame(self, env_ids)


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
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to identity."""

    class_type: type = SequentialPoseCommand

    asset_name: str = MISSING  # type: ignore
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING  # type: ignore
    """Name of the end-effector body used for pose commands."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    commands: Sequence[BaseCmd] = MISSING  # type: ignore
    """The sequence of commands to execute."""

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

    debug_vis: bool = False

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
    canonicalize_upward: bool = True
    """If True, flip the target 180° about TCP +X when TCP +Z points down.

    That assumes Spot's TCP (+Z up, +X approach). Franka TCP is +Z approach,
    +X hand-top — leave this False there so the authored frame rotation is
    used as-is.
    """


@configclass
class CuroboPlannedGoToFrameCfg(GoToFrameCfg):
    """A cuRobo-planned variant of :class:`GoToFrameCfg`."""

    class_type = _CuroboPlannedGoToFrameHandler
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    interpolation_buffer_size: int = 128


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
    axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    """Rotation axis expressed in ``target_frame_name`` coordinates."""
    angle_deg: float = -90.0
    """Angle in degrees to rotate around :attr:`axis`."""
    angular_velocity: float = 0.3
    """Angular speed of the commanded arc [rad/s]."""
    gripper_open: bool = False
    """Status of the gripper during the command."""
    angle_threshold_deg: float = 5.0
    """Remaining commanded arc at which the command is done [deg]."""


@configclass
class CuroboPlannedRotateFrameCfg(RotateFrameCfg):
    """A cuRobo-planned endpoint variant of :class:`RotateFrameCfg`."""

    class_type = _CuroboPlannedRotateFrameHandler
    robot_joint_names: list[str] = MISSING  # type: ignore
    robot_curobo_yaml: str | None = None
    robot_urdf: str | None = None
    num_ik_seeds: int = 4
    interpolation_buffer_size: int = 128
