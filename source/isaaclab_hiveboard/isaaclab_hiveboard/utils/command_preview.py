# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic previews using the task's command geometry and robot Jacobians."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    _GoToFrameHandler,
    _RotateFrameHandler,
    _ScrewFrameHandler,
)
from isaaclab_hiveboard.utils.frame_sensors import refresh_frame_sensors as refresh_frame_sensors


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def numpy(value):
    return tensor(value).detach().cpu().numpy()


class _PreviewContext:
    """Let production handlers resolve geometry from an authored starting pose."""

    def __init__(self, term, pose):
        self.term = term
        self.pose = pose
        self.cfg = copy.copy(term.cfg)
        self.cfg.debug_vis = False

    def __getattr__(self, name):
        return getattr(self.term, name)

    def _get_ee_in_base_frame(self, env_ids):
        return self.pose


@dataclass
class Segment:
    cfg: object
    start: tuple
    end: tuple
    duration: float
    handler: object = None
    pivot: tuple | None = None

    @property
    def gripper_open(self):
        return self.cfg.open_gripper if isinstance(self.cfg, GripperCommand) else self.cfg.gripper_open

    def sample(self, fraction: float):
        fraction = min(1.0, max(0.0, fraction))
        if isinstance(self.cfg, GripperCommand):
            return self.start
        p0, q0 = self.start
        if isinstance(self.cfg, RotateFrameCfg):
            h = self.handler
            angle = h.angle_rad_tensor * fraction
            pos = h.axis_pos_b + h._rodrigues_rotate(h.radius_vec, h.rot_axis_b, angle)
            if isinstance(self.cfg, ScrewFrameCfg):
                pos += h.rot_axis_b * (fraction * self.cfg.axial_distance)
            quat = math_utils.quat_mul(math_utils.quat_from_angle_axis(angle, h.rot_axis_b), q0)
            return pos, quat
        reference = getattr(self.cfg, "reference_pos_env", None)
        if reference is not None:
            # Retargeted commands follow their dense authored reference.
            position = fraction * (len(reference) - 1)
            index = min(int(position), len(reference) - 2)
            weight = position - index
            env = self.handler._command_term._env
            pos = p0.new_tensor(reference[index]) * (1 - weight) + p0.new_tensor(reference[index + 1]) * weight
            qs = self.cfg.reference_quat_xyzw
            quat0, quat1 = q0.new_tensor([qs[index]]), q0.new_tensor([qs[index + 1]])
            quat = math_utils.quat_box_plus(quat0, math_utils.quat_box_minus(quat1, quat0) * weight)
            robot = self.handler._asset
            return math_utils.subtract_frame_transforms(
                tensor(robot.data.root_pos_w),
                tensor(robot.data.root_quat_w),
                pos[None] + env.scene.env_origins,
                quat,
            )
        p1, q1 = self.end
        elapsed = fraction * self.duration
        delta = p1 - p0
        distance = float(torch.linalg.vector_norm(delta))
        pos = p0 + delta * min(1.0, elapsed * self.cfg.velocity / max(distance, 1e-8))
        error = math_utils.quat_box_minus(q1, q0)
        angle = float(torch.linalg.vector_norm(error))
        quat = math_utils.quat_box_plus(q0, error * min(1.0, elapsed * self.cfg.angular_velocity / max(angle, 1e-8)))
        return pos, quat


def build_segments(term, commands, initial_pose) -> list[Segment]:
    """Resolve goals and pivot frames with the same handlers used by play.py.

    cuRobo variants use their Cartesian geometry here; physical/planned replay
    remains the responsibility of play.py. Object poses are the current snapshot.
    """
    context = _PreviewContext(term, initial_pose)
    ids = torch.tensor([0], device=term.device)
    segments = []
    for cfg in commands:
        start = context.pose
        if isinstance(cfg, GripperCommand):
            segment = Segment(cfg, start, start, cfg.duration_s)
        elif isinstance(cfg, GoToFrameCfg):
            handler = _GoToFrameHandler(cfg, context)
            handler.reset(ids)
            end = handler.get_target_in_base_frame(ids)
            duration = max(
                float(torch.linalg.vector_norm(end[0] - start[0])) / cfg.velocity,
                float(math_utils.quat_error_magnitude(end[1], start[1])[0]) / cfg.angular_velocity,
                0.05,
            )
            segment = Segment(cfg, start, end, duration, handler)
        elif isinstance(cfg, RotateFrameCfg):
            handler_cls = _ScrewFrameHandler if isinstance(cfg, ScrewFrameCfg) else _RotateFrameHandler
            handler = handler_cls(cfg, context)
            handler.reset(ids)
            end = handler.get_target_in_base_frame(ids)
            duration = max(abs(float(handler.angle_rad_tensor[0])) / cfg.angular_velocity, 0.05)
            segment = Segment(cfg, start, end, duration, handler, handler._get_rotation_axis_pose_b(ids))
        else:
            raise ValueError(f"Unsupported command: {type(cfg).__name__}")
        if not bool(torch.isfinite(segment.end[0]).all()) or not bool(torch.isfinite(segment.end[1]).all()):
            raise ValueError("Reference sensor produced a non-finite goal pose")
        if float(torch.linalg.vector_norm(segment.end[1])) < 1e-6:
            raise ValueError("Reference sensor has not produced a valid quaternion; refresh frame sensors after reset")
        segments.append(segment)
        context.pose = segment.end
    return segments


class PreviewIK:
    """Warm-started DLS with joint limits; writes kinematic states, never steps physics."""

    def __init__(self, env, term):
        self.env = env
        self.term = term
        self.robot = env.scene[term.cfg.asset_name]
        self.body_idx = term._body_idx
        arm = env.action_manager.get_term("arm_action")
        self.arm_ids = list(self.robot.find_joints(arm.cfg.joint_names, preserve_order=True)[0])
        if not self.arm_ids:
            raise ValueError("arm_action does not identify any joints")
        self.gripper = (
            env.action_manager.get_term("gripper_action")
            if "gripper_action" in env.action_manager.active_terms
            else None
        )
        self.initial_q = tensor(self.robot.data.joint_pos).clone()
        self.initial_flange = (
            tensor(self.robot.data.body_pos_w)[:, self.body_idx].clone(),
            tensor(self.robot.data.body_quat_w)[:, self.body_idx].clone(),
        )
        self.offset = copy.deepcopy(term.cfg.body_offset)
        self.jacobian_body = self.body_idx - 1 if self.robot.is_fixed_base else self.body_idx
        self.jacobian_joints = [j + int(self.robot.num_base_dofs) for j in self.arm_ids]
        limits = tensor(self.robot.data.joint_pos_limits)[0, self.arm_ids]
        self.lo, self.hi = limits[:, 0], limits[:, 1]

    def _with_offset(self, pos, quat):
        if self.offset is None:
            return pos, quat
        return math_utils.combine_frame_transforms(
            pos,
            quat,
            pos.new_tensor([self.offset.pos]),
            quat.new_tensor([self.offset.rot]),
        )

    def initial_pose_b(self):
        return self.to_base(*self._with_offset(*self.initial_flange))

    def to_base(self, pos, quat):
        return math_utils.subtract_frame_transforms(
            tensor(self.robot.data.root_pos_w),
            tensor(self.robot.data.root_quat_w),
            pos,
            quat,
        )

    def to_world(self, pos, quat):
        return math_utils.combine_frame_transforms(
            tensor(self.robot.data.root_pos_w),
            tensor(self.robot.data.root_quat_w),
            pos,
            quat,
        )

    def tcp_pose_w(self):
        return self._with_offset(
            tensor(self.robot.data.body_pos_w)[:, self.body_idx],
            tensor(self.robot.data.body_quat_w)[:, self.body_idx],
        )

    def write(self, q):
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.update(0.0)

    def reset(self):
        self.write(self.initial_q.clone())

    def solve(self, pose_b, gripper_open: bool, iters: int = 24) -> tuple[float, float]:
        goal_pos, goal_quat = self.to_world(*pose_b)
        q = tensor(self.robot.data.joint_pos).clone()
        if self.gripper is not None:
            # Resolve the actual task's gripper expressions, including mimic joints.
            grip = self.gripper._open_command if gripper_open else self.gripper._close_command
            q[:, self.gripper._joint_ids] = grip
        self.write(q)
        for _ in range(iters):
            pos, quat = self.tcp_pose_w()
            error = torch.cat((goal_pos - pos, math_utils.quat_box_minus(goal_quat, quat)), dim=-1)
            if (
                float(torch.linalg.vector_norm(error[:, :3])) < 0.001
                and float(torch.linalg.vector_norm(error[:, 3:])) < 0.01
            ):
                break
            jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body, :, self.jacobian_joints].clone()
            if self.offset is not None:
                offset_w = math_utils.quat_apply(
                    tensor(self.robot.data.body_quat_w)[:, self.body_idx],
                    pos.new_tensor([self.offset.pos]),
                )
                jac[:, :3] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_w), jac[:, 3:])
            lhs = jac @ jac.transpose(1, 2) + torch.eye(6, device=q.device)[None] * 1e-4
            delta = (jac.transpose(1, 2) @ torch.linalg.solve(lhs, error[..., None]))[..., 0]
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("IK produced a non-finite joint update")
            q[:, self.arm_ids] = torch.clamp(q[:, self.arm_ids] + delta.clamp(-0.15, 0.15), self.lo, self.hi)
            self.write(q)
        # Measure the final written state, including after the last iteration.
        pos, quat = self.tcp_pose_w()
        return (
            float(torch.linalg.vector_norm(goal_pos - pos)),
            math.degrees(float(math_utils.quat_error_magnitude(goal_quat, quat)[0])),
        )


def pose_to_viser(pose) -> dict:
    pos, quat = pose
    return {"position": numpy(pos)[0], "wxyz": numpy(quat)[0, [3, 0, 1, 2]]}


def viser_to_pose(position, wxyz, *, device):
    quat = np.asarray(wxyz)[[1, 2, 3, 0]]
    return (
        torch.tensor(np.asarray(position)[None], dtype=torch.float32, device=device),
        torch.tensor(quat[None], dtype=torch.float32, device=device),
    )
