# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kinematic previews using the task's command geometry and robot Jacobians."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    _CuroboPlannedGoToFrameHandler,
    _CuroboPlannedRotateFrameHandler,
    _GoToFrameHandler,
    _RotateFrameHandler,
    _ScrewFrameHandler,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
)
from isaaclab_hiveboard.utils.frame_sensors import refresh_frame_sensors as refresh_frame_sensors


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def _fmt_xyz(value) -> str:
    vec = numpy(value).reshape(-1)[:3]
    return "[" + ", ".join(f"{float(x):.4f}" for x in vec) + "]"


def _fmt_rpy_deg(quat) -> str:
    q = tensor(quat)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    rpy = torch.rad2deg(torch.stack(math_utils.euler_xyz_from_quat(q), dim=-1))[0]
    return "[" + ", ".join(f"{float(x):.1f}" for x in rpy) + "]"


def _fmt_axis(quat, vec) -> str:
    q = tensor(quat)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    axis = math_utils.quat_apply(q, q.new_tensor([list(vec)]))[0]
    return "[" + ", ".join(f"{float(x):.3f}" for x in axis) + "]"


def format_ik_debug(report: dict) -> str:
    return "\n".join(f"{key}: {value}" for key, value in report.items())


def apply_authored_command_field(cfg, name: str, value) -> bool:
    """Write a GUI field onto a command.

    Editing ``angle_deg`` turns off ``use_valve_angle`` so the typed angle is
    the arc that preview and play execute. Returns True when sibling fields
    changed and the editor panel should refresh.
    """
    setattr(cfg, name, value)
    if name == "angle_deg" and getattr(cfg, "use_valve_angle", False):
        cfg.use_valve_angle = False
        return True
    return False


def effective_arc_caption(cfg, handler) -> str:
    """Explain which authored field produced ``handler.angle_rad_tensor``."""
    deg = math.degrees(float(handler.angle_rad_tensor[0]))
    authored = float(getattr(cfg, "angle_deg", 0.0))
    if getattr(cfg, "use_valve_angle", False):
        source = f"from remaining valve angle; Angle (deg)={authored:g} is unused"
    else:
        source = "from Angle (deg)"
    limit = float(getattr(cfg, "max_ee_rotation_deg", 0.0) or 0.0)
    if limit > 0.0:
        source += f", clamped to ±{limit:g}°"
    return f"Effective arc: **{deg:.1f}°** ({source})"


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

    def sample(self, fraction: float, *, use_cached_plan: bool = True):
        fraction = min(1.0, max(0.0, fraction))
        if isinstance(self.cfg, GripperCommand):
            return self.start
        plan = getattr(self.cfg, "_preview_plan", None) if use_cached_plan else None
        if plan is not None:
            return _interp_pose(plan["tcp_pos_b"], plan["tcp_quat_b"], fraction)
        p0, q0 = self.start
        if isinstance(self.cfg, RotateFrameCfg):
            h = self.handler
            angle = h.angle_rad_tensor * fraction
            pos = h.axis_pos_b + h.axial_vec + h._rodrigues_rotate(h.radius_vec, h.rot_axis_b, angle)
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

    def sample_joints(self, fraction: float):
        plan = getattr(self.cfg, "_preview_plan", None)
        if plan is None:
            return None
        return _interp_vector(plan["joints"], fraction), plan["joint_names"]


def _interp_vector(values: torch.Tensor, fraction: float) -> torch.Tensor:
    fraction = min(1.0, max(0.0, fraction))
    count = values.shape[0]
    if count == 1:
        return values[0]
    position = fraction * (count - 1)
    index = min(int(position), count - 2)
    weight = position - index
    return values[index] * (1.0 - weight) + values[index + 1] * weight


def _interp_pose(pos: torch.Tensor, quat: torch.Tensor, fraction: float):
    value = _interp_vector(pos, fraction).unsqueeze(0)
    if pos.shape[0] == 1:
        return value, quat[0:1]
    position = min(1.0, max(0.0, fraction)) * (pos.shape[0] - 1)
    index = min(int(position), pos.shape[0] - 2)
    weight = position - index
    q = math_utils.quat_box_plus(
        quat[index : index + 1],
        math_utils.quat_box_minus(quat[index + 1 : index + 2], quat[index : index + 1]) * weight,
    )
    return value, q


def _preview_waypoint_count(segment: Segment, dt: float) -> int:
    cfg = segment.cfg
    if isinstance(cfg, GripperCommand):
        return 1
    if isinstance(cfg, RotateFrameCfg) and segment.handler is not None:
        angle = abs(float(segment.handler.angle_rad_tensor[0]))
        step = max(float(cfg.angular_velocity) * dt, 1e-3)
        return int(min(48, max(2, math.ceil(angle / step))))
    return int(min(16, max(2, math.ceil(float(segment.duration) / 0.08))))


def invalidate_preview_plans_from(commands, index: int = 0) -> None:
    """Drop cached cuRobo polylines from ``index`` onward.

    Later legs store TCP waypoints that start at the previous command's old
    endpoint. After an edit those paths no longer chain; rebuild must sample
    live geometry instead of the stale plan.
    """
    for cfg in commands[index:]:
        if hasattr(cfg, "_preview_plan"):
            cfg._preview_plan = None


def _stacked_tcp(segment: Segment, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    poses = [
        segment.sample(0.0 if count == 1 else i / (count - 1), use_cached_plan=False)
        for i in range(count)
    ]
    pos = torch.cat([p for p, _ in poses], dim=0)
    quat = torch.cat([q for _, q in poses], dim=0)
    return pos, quat


def concat_segment_tcp(segments: list[Segment], dt: float) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Join each command's Cartesian path, dropping duplicate endpoints."""
    chunks_pos: list[torch.Tensor] = []
    chunks_quat: list[torch.Tensor] = []
    ranges: list[tuple[int, int]] = []
    for segment in segments:
        pos, quat = _stacked_tcp(segment, _preview_waypoint_count(segment, dt))
        if chunks_pos:
            same_pos = torch.allclose(pos[0], chunks_pos[-1][-1], atol=1e-3)
            same_quat = float(math_utils.quat_error_magnitude(quat[0:1], chunks_quat[-1][-1:])) < 0.05
            if same_pos and same_quat:
                pos, quat = pos[1:], quat[1:]
            if pos.shape[0] == 0:
                last = sum(chunk.shape[0] for chunk in chunks_pos) - 1
                ranges.append((last, last + 1))
                continue
        start = sum(chunk.shape[0] for chunk in chunks_pos)
        ranges.append((start, start + pos.shape[0]))
        chunks_pos.append(pos)
        chunks_quat.append(quat)
    return torch.cat(chunks_pos, dim=0), torch.cat(chunks_quat, dim=0), ranges


def _retarget_tcp_sequence(term, tcp_pos_b: torch.Tensor, tcp_quat_b: torch.Tensor, planner: dict) -> torch.Tensor:
    """One MotionRetargeter pass over a flange path. Returns (T, dof) arm joints."""
    from isaaclab_hiveboard.assets import ASSET_DIR
    from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
    from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

    with curobo_compatible_warp():
        from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg, SequenceGoalToolPose
        from curobo.types import DeviceCfg, JointState, ToolPoseCriteria

        robot_cfg = load_curobo_robot_cfg(
            planner.get("robot_curobo_yaml") or f"{ASSET_DIR}/spot/cumotion/spot_arm.yaml",
            planner.get("robot_urdf") or f"{ASSET_DIR}/spot/spot_with_arm.urdf",
        )
        tool_frame = robot_cfg["robot_cfg"]["kinematics"]["tool_frames"][0]
        cache_key = (
            "preview_retargeter",
            planner.get("robot_curobo_yaml"),
            planner.get("robot_urdf"),
            planner.get("num_ik_seeds", 4),
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
                        optimization_dt=float(term._env.step_dt),
                        num_seeds_global=int(planner.get("num_ik_seeds", 4)),
                        num_seeds_local=1,
                        position_tolerance=0.002,
                        orientation_tolerance=0.02,
                        device_cfg=DeviceCfg(),
                    )
                )

        retargeter, build_s = term.get_curobo_solver(cache_key, _make_retargeter)
        solve_start = time.perf_counter()
        env_ids = torch.tensor([0], device=term.device, dtype=torch.long)
        flange_pos_b, flange_quat_b = term._tcp_pose_to_body_pose(tcp_pos_b, tcp_quat_b, env_ids)
        joint_names = list(planner["robot_joint_names"])
        joint_ids, _ = term._asset.find_joints(joint_names, preserve_order=True)
        current = JointState.from_position(
            term._asset.data.joint_pos.torch[env_ids][:, joint_ids],
            joint_names=joint_names,
        )
        curobo_flange = retargeter.kinematics.compute_kinematics(current).tool_poses.get_link_pose(tool_frame)
        isaac_flange_pos_b, isaac_flange_quat_b = math_utils.subtract_frame_transforms(
            term._asset.data.root_pos_w.torch[env_ids],
            term._asset.data.root_quat_w.torch[env_ids],
            term._asset.data.body_pos_w.torch[env_ids, term._body_idx],
            term._asset.data.body_quat_w.torch[env_ids, term._body_idx],
        )
        flange_quat_inv_c = math_utils.quat_inv(_wxyz_to_xyzw(curobo_flange.quaternion))
        flange_pos_inv_c = -math_utils.quat_apply(flange_quat_inv_c, curobo_flange.position)
        curobo_base_pos_b, curobo_base_quat_b = math_utils.combine_frame_transforms(
            isaac_flange_pos_b, isaac_flange_quat_b, flange_pos_inv_c, flange_quat_inv_c
        )
        n = tcp_pos_b.shape[0]
        flange_pos_c, flange_quat_c = math_utils.subtract_frame_transforms(
            curobo_base_pos_b.expand(n, -1),
            curobo_base_quat_b.expand(n, -1),
            flange_pos_b,
            flange_quat_b,
        )
        with torch.inference_mode(False), torch.enable_grad():
            current = JointState.from_position(current.position.clone(), joint_names=joint_names)
            targets = SequenceGoalToolPose(
                tool_frames=[tool_frame],
                position=flange_pos_c[:, None, None, None, :].clone(),
                quaternion=_xyzw_to_wxyz(flange_quat_c[:, None, None, None, :].clone()),
            )
            if hasattr(retargeter, "_set_initial_joint_state"):
                result = retargeter.solve_sequence(targets, initial_joint_state=current)
                joints = result.joint_state.reorder(joint_names).position
            else:
                retargeter.reset()
                retargeter._prev_solution = current.position.clone()
                retargeter._prev_velocity = current.velocity.clone()
                joints = torch.stack(
                    [
                        retargeter.solve_frame(targets.get_frame(index)).joint_state.reorder(joint_names).position
                        for index in range(targets.num_frames)
                    ],
                    dim=1,
                )
        if joints.ndim == 3:
            joints = joints[0]
        print(
            f"[EDITOR] retargeted {n} TCP waypoints in {time.perf_counter() - solve_start:.2f}s "
            f"(solver build {build_s:.2f}s)",
            flush=True,
        )
        return joints


def format_infeasible_rotation_message(err: BaseException, handler=None) -> str:
    """Editor-facing summary: the rotate arc cannot be tracked as a joint path."""
    header = str(err).strip().splitlines()[0]
    if header.startswith("Rotation is infeasible"):
        return str(err).strip()
    angle_txt = ""
    if handler is not None and getattr(handler, "angle_rad_tensor", None) is not None:
        angle_txt = f" Commanded arc is {math.degrees(float(handler.angle_rad_tensor[0])):.1f}°."
    return (
        "Rotation is infeasible. "
        f"{header}"
        f"{angle_txt} "
        "The wrist cannot follow this arc as a continuous joint path. "
        "Reduce the rotation (max_ee_rotation_deg or angle_deg, or disable use_valve_angle) and plan again."
    )


def format_editor_plan_failure(err: BaseException) -> str:
    """Markdown status for a failed Plan-with-cuRobo click."""
    text = str(err).strip()
    first = text.splitlines()[0] if text else "unknown error"
    if "infeasible" in text.lower():
        body = first if first.startswith("Rotation is infeasible") else format_infeasible_rotation_message(err)
        return f"**Rotation is infeasible**\n\n{body}"
    return f"**cuRobo planning failed:** {first}"


def plan_curobo_segments(term, segments: list[Segment], ik: "PreviewIK") -> int:
    """Retarget the Cartesian sequence, then re-plan rotates with the play handler.

    Whole-sequence retarget is a coarse preview. Rotate feasibility must use the
    same dense local-IK arc as play.py; an infeasible wrist path raises.
    """
    from isaaclab_hiveboard.utils.command_setup import planner_settings

    ik.reset()
    try:
        planner = planner_settings([segment.cfg for segment in segments] + list(term.cfg.commands))
    except ValueError:
        print("[EDITOR] no cuRobo planner settings; keeping DLS preview", flush=True)
        return 0
    dt = float(term._env.step_dt)
    try:
        tcp_pos, tcp_quat, ranges = concat_segment_tcp(segments, dt)
        joints = _retarget_tcp_sequence(term, tcp_pos, tcp_quat, planner)
    except Exception as err:
        print(f"[EDITOR] whole-sequence retarget failed ({err}); falling back per command", flush=True)
        return _plan_curobo_per_segment(term, segments, ik)
    names = list(planner["robot_joint_names"])
    planned = 0
    for segment, (start, end) in zip(segments, ranges):
        segment.cfg._preview_plan = {
            "joints": joints[start:end].detach(),
            "joint_names": names,
            "tcp_pos_b": tcp_pos[start:end].detach(),
            "tcp_quat_b": tcp_quat[start:end].detach(),
        }
        planned += 1
    _plan_curobo_rotates_like_play(term, segments, ik)
    ik.reset()
    return planned


def _write_segment_end(ik: "PreviewIK", segment: Segment) -> None:
    if isinstance(segment.cfg, GripperCommand):
        ik.write(ik._gripper_q(tensor(ik.robot.data.joint_pos).clone(), segment.gripper_open))
        return
    joints = segment.sample_joints(1.0)
    if joints is not None:
        q_arm, names = joints
        ik.apply_named_joints(names, q_arm, segment.gripper_open)
        return
    ik.solve(segment.end, segment.gripper_open)


def _plan_rotate_like_play(term, cfg, ik: "PreviewIK") -> None:
    """Plan one rotate with the play.py handler. Raises if the joint arc is infeasible."""
    ids = torch.tensor([0], device=term.device, dtype=torch.long)
    handler = _CuroboPlannedRotateFrameHandler(cfg, term)
    handler.reset(ids)
    try:
        handler._plan(ids)
    except RuntimeError as err:
        cfg._preview_plan = None
        raise RuntimeError(format_infeasible_rotation_message(err, handler)) from err
    if handler._fallback or handler._joint_waypoints is None:
        cfg._preview_plan = None
        raise RuntimeError(
            format_infeasible_rotation_message(
                RuntimeError("cuRobo rotate planner did not produce a joint arc"),
                handler,
            )
        )
    wp = handler._joint_waypoints
    if wp.ndim == 3:
        wp = wp[0]
    cfg._preview_plan = {
        "joints": wp.detach(),
        "joint_names": list(cfg.robot_joint_names),
        "tcp_pos_b": handler._waypoint_pos_b[0].detach(),
        "tcp_quat_b": handler._waypoint_quat_b[0].detach(),
    }
    ik.apply_named_joints(list(cfg.robot_joint_names), wp[-1], cfg.gripper_open)


def _plan_curobo_rotates_like_play(term, segments: list[Segment], ik: "PreviewIK") -> None:
    """Replace coarse rotate previews with play-density plans; raise if infeasible."""
    ik.reset()
    for segment in segments:
        if isinstance(segment.cfg, CuroboPlannedRotateFrameCfg):
            _plan_rotate_like_play(term, segment.cfg, ik)
            continue
        _write_segment_end(ik, segment)


def _plan_curobo_per_segment(term, segments: list[Segment], ik: "PreviewIK") -> int:
    """Fallback: plan each CuroboPlanned command with its production handler."""
    from isaaclab_hiveboard.utils.command_setup import is_curobo_command

    planned = 0
    ik.reset()
    ids = torch.tensor([0], device=term.device, dtype=torch.long)
    for segment in segments:
        cfg = segment.cfg
        if isinstance(cfg, GripperCommand):
            ik.write(ik._gripper_q(tensor(ik.robot.data.joint_pos).clone(), segment.gripper_open))
            continue
        if isinstance(cfg, CuroboPlannedRotateFrameCfg):
            _plan_rotate_like_play(term, cfg, ik)
            planned += 1
            continue
        if not is_curobo_command(cfg):
            ik.solve(segment.end, segment.gripper_open)
            continue
        handler = _CuroboPlannedGoToFrameHandler(cfg, term)
        try:
            handler.reset(ids)
            handler._plan(ids)
        except Exception as err:
            print(f"[EDITOR] cuRobo plan failed for {type(cfg).__name__}: {err}", flush=True)
            cfg._preview_plan = None
            ik.solve(segment.end, segment.gripper_open)
            continue
        if handler._fallback or handler._joint_waypoints is None:
            print(f"[EDITOR] cuRobo fell back to servo for {type(cfg).__name__}", flush=True)
            cfg._preview_plan = None
            ik.solve(segment.end, segment.gripper_open)
            continue
        wp = handler._joint_waypoints
        if wp.ndim == 3:
            wp = wp[0]
        cfg._preview_plan = {
            "joints": wp.detach(),
            "joint_names": list(cfg.robot_joint_names),
            "tcp_pos_b": handler._waypoint_pos_b[0].detach(),
            "tcp_quat_b": handler._waypoint_quat_b[0].detach(),
        }
        q = tensor(ik.robot.data.joint_pos).clone()
        joint_ids, _ = ik.robot.find_joints(list(cfg.robot_joint_names), preserve_order=True)
        q[:, joint_ids] = wp[-1].to(device=q.device, dtype=q.dtype)
        ik.write(ik._gripper_q(q, segment.gripper_open))
        planned += 1
    ik.reset()
    return planned


def build_segments(term, commands, initial_pose) -> list[Segment]:
    """Resolve goals and pivot frames with the same handlers used by play.py.

    Cartesian geometry is always available. ``plan_curobo_segments`` overlays
    cuRobo joint/TCP waypoints onto CuroboPlanned commands when requested.
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
        self.arm_ids, self.arm_names = self.robot.find_joints(arm.cfg.joint_names, preserve_order=True)
        self.arm_ids = list(self.arm_ids)
        self.arm_names = list(self.arm_names)
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
        self.last_debug: dict[str, object] = {}

    def apply_named_joints(self, names: list[str], values: torch.Tensor, gripper_open: bool) -> None:
        joint_ids, _ = self.robot.find_joints(list(names), preserve_order=True)
        q = tensor(self.robot.data.joint_pos).clone()
        q[:, joint_ids] = values.to(device=q.device, dtype=q.dtype).reshape(1, -1)
        self.write(self._gripper_q(q, gripper_open))

    def measure(self, pose_b) -> tuple[float, float]:
        """Record TCP/flange residuals for the current written pose. Does not solve."""
        tcp_goal_pos, tcp_goal_quat = self.to_world(*pose_b)
        goal_pos, goal_quat = self._without_offset(tcp_goal_pos, tcp_goal_quat)
        tcp_pos, tcp_quat = self.tcp_pose_w()
        flange_pos, flange_quat = self._flange_pose_w()
        pos_error = float(torch.linalg.vector_norm(tcp_goal_pos - tcp_pos))
        rot_error = math.degrees(float(math_utils.quat_error_magnitude(tcp_goal_quat, tcp_quat)[0]))
        jac = tensor(self.robot.data.body_link_jacobian_w)
        arm_q = tensor(self.robot.data.joint_pos)[0, self.arm_ids]
        names = list(getattr(self.robot, "body_names", []))
        self.last_debug = {
            "body": getattr(self.term.cfg, "body_name", "?"),
            "body_idx": int(self.body_idx),
            "body_name_at_idx": names[self.body_idx] if 0 <= self.body_idx < len(names) else "?",
            "fixed_base": bool(self.robot.is_fixed_base),
            "num_bodies": int(getattr(self.robot, "num_bodies", len(names) or -1)),
            "num_base_dofs": int(self.robot.num_base_dofs),
            "jacobian_body": int(self.jacobian_body),
            "jacobian_shape": tuple(int(v) for v in jac.shape),
            "arm_joint_names": self.arm_names,
            "jacobian_joints": [int(j) for j in self.jacobian_joints],
            "offset_pos": tuple(float(v) for v in (self.offset.pos if self.offset else (0, 0, 0))),
            "offset_rot_xyzw": tuple(float(v) for v in (self.offset.rot if self.offset else (0, 0, 0, 1))),
            "offset_rpy_deg": _fmt_rpy_deg(
                tcp_quat.new_tensor([self.offset.rot if self.offset else (0, 0, 0, 1)])
            ),
            "arm_q_deg": [round(math.degrees(float(v)), 1) for v in arm_q],
            "tcp_goal_pos": _fmt_xyz(tcp_goal_pos),
            "tcp_goal_rpy_deg": _fmt_rpy_deg(tcp_goal_quat),
            "tcp_goal_+X": _fmt_axis(tcp_goal_quat, (1, 0, 0)),
            "tcp_goal_+Z": _fmt_axis(tcp_goal_quat, (0, 0, 1)),
            "tcp_actual_pos": _fmt_xyz(tcp_pos),
            "tcp_actual_rpy_deg": _fmt_rpy_deg(tcp_quat),
            "tcp_actual_+X": _fmt_axis(tcp_quat, (1, 0, 0)),
            "tcp_actual_+Z": _fmt_axis(tcp_quat, (0, 0, 1)),
            "flange_goal_pos": _fmt_xyz(goal_pos),
            "flange_goal_rpy_deg": _fmt_rpy_deg(goal_quat),
            "flange_actual_pos": _fmt_xyz(flange_pos),
            "flange_actual_rpy_deg": _fmt_rpy_deg(flange_quat),
            "flange_actual_+X": _fmt_axis(flange_quat, (1, 0, 0)),
            "flange_actual_+Z": _fmt_axis(flange_quat, (0, 0, 1)),
            "pos_error_mm": round(pos_error * 1000.0, 2),
            "rot_error_deg": round(rot_error, 2),
        }
        return pos_error, rot_error

    def _offset_tensors(self, like: torch.Tensor):
        return like.new_tensor([self.offset.pos]), like.new_tensor([self.offset.rot])

    def _with_offset(self, pos, quat):
        if self.offset is None:
            return pos, quat
        return math_utils.combine_frame_transforms(pos, quat, *self._offset_tensors(pos))

    def _without_offset(self, pos, quat):
        """Map a TCP pose back to the flange (``arm_link_wr1``)."""
        if self.offset is None:
            return pos, quat
        offset_pos, offset_rot = self._offset_tensors(pos)
        flange_from_tcp_rot = math_utils.quat_inv(offset_rot)
        flange_from_tcp_pos = -math_utils.quat_apply(flange_from_tcp_rot, offset_pos)
        return math_utils.combine_frame_transforms(pos, quat, flange_from_tcp_pos, flange_from_tcp_rot)

    def _flange_pose_w(self):
        return (
            tensor(self.robot.data.body_pos_w)[:, self.body_idx],
            tensor(self.robot.data.body_quat_w)[:, self.body_idx],
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

    def _gripper_q(self, q: torch.Tensor, gripper_open: bool) -> torch.Tensor:
        if self.gripper is None:
            return q
        q = q.clone()
        grip = self.gripper._open_command if gripper_open else self.gripper._close_command
        q[:, self.gripper._joint_ids] = grip
        return q

    def _arm_seeds(self, gripper_open: bool) -> list[torch.Tensor]:
        """Stowed pose plus a mid-range arm; DLS from a limit-saturated wrist cannot recover."""
        stowed = self._gripper_q(self.initial_q.clone(), gripper_open)
        mid = stowed.clone()
        mid[:, self.arm_ids] = 0.5 * (self.lo + self.hi)
        current = self._gripper_q(tensor(self.robot.data.joint_pos).clone(), gripper_open)
        return [stowed, mid, current]

    def _dls(self, q: torch.Tensor, goal_pos: torch.Tensor, goal_quat: torch.Tensor, iters: int) -> torch.Tensor:
        self.write(q)
        for _ in range(iters):
            pos, quat = self._flange_pose_w()
            error = torch.cat((goal_pos - pos, math_utils.quat_box_minus(goal_quat, quat)), dim=-1)
            if (
                float(torch.linalg.vector_norm(error[:, :3])) < 0.001
                and float(torch.linalg.vector_norm(error[:, 3:])) < 0.01
            ):
                break
            jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.jacobian_body, :, self.jacobian_joints].clone()
            lhs = jac @ jac.transpose(1, 2) + torch.eye(6, device=q.device)[None] * 1e-4
            delta = (jac.transpose(1, 2) @ torch.linalg.solve(lhs, error[..., None]))[..., 0]
            if not bool(torch.isfinite(delta).all()):
                raise ValueError("IK produced a non-finite joint update")
            q = q.clone()
            q[:, self.arm_ids] = torch.clamp(q[:, self.arm_ids] + delta.clamp(-0.15, 0.15), self.lo, self.hi)
            self.write(q)
        return q

    def solve(self, pose_b, gripper_open: bool, iters: int = 24) -> tuple[float, float]:
        # Commands are TCP poses. cuRobo / play.py track arm_link_wr1, so preview
        # IK at the flange with the inverse offset instead of a TCP Jacobian.
        tcp_goal_pos, tcp_goal_quat = self.to_world(*pose_b)
        goal_pos, goal_quat = self._without_offset(tcp_goal_pos, tcp_goal_quat)
        best_q, best_cost = None, float("inf")
        for seed in self._arm_seeds(gripper_open):
            q = self._dls(seed, goal_pos, goal_quat, iters)
            tcp_pos, tcp_quat = self.tcp_pose_w()
            pos_error = float(torch.linalg.vector_norm(tcp_goal_pos - tcp_pos))
            rot_error = math.degrees(float(math_utils.quat_error_magnitude(tcp_goal_quat, tcp_quat)[0]))
            cost = pos_error + 0.005 * rot_error
            if cost < best_cost:
                best_q, best_cost = q, cost
            if pos_error < 0.005 and rot_error < 3.0:
                break
        self.write(best_q)
        return self.measure(pose_b)


def pose_to_viser(pose) -> dict:
    pos, quat = pose
    return {"position": numpy(pos)[0], "wxyz": numpy(quat)[0, [3, 0, 1, 2]]}


def viser_to_pose(position, wxyz, *, device):
    quat = np.asarray(wxyz)[[1, 2, 3, 0]]
    return (
        torch.tensor(np.asarray(position)[None], dtype=torch.float32, device=device),
        torch.tensor(quat[None], dtype=torch.float32, device=device),
    )
