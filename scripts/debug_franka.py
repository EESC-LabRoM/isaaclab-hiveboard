# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless FR3 play that dumps tracking / chaos logs for later inspection.

Writes ``logs/franka_debug/<task>_<stamp>/``:
  - ``traj.csv``       per-step TCP, command, root, joints
  - ``summary.json``   pass/fail checks and extrema
  - ``summary.txt``    the same in plain text

Example:
  uv run python scripts/debug_franka.py --headless --device cuda:0
  uv run python scripts/debug_franka.py --position-only --max-steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Log Franka FR3 tracking metrics.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-HiveBoard-Franka-LeverValve-v0",
    help="Gym task id (Franka only).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="Output directory (default: logs/franka_debug/<task>_<stamp>).",
)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument(
    "--position-only",
    action="store_true",
    default=False,
    help="Disable the IK orientation objective (translation-only ablation).",
)
parser.add_argument(
    "--log-every",
    type=int,
    default=20,
    help="Print a compact line every N env steps (0 disables).",
)
parser.add_argument(
    "--target-pitch-deg",
    type=float,
    default=None,
    help="Temporary circuit-breaker target-frame pitch for IK reachability sweeps.",
)
parser.add_argument(
    "--target-roll-deg",
    type=float,
    default=0.0,
    help="Temporary circuit-breaker target-frame roll for IK reachability sweeps.",
)
parser.add_argument(
    "--above-pitch-deg",
    type=float,
    default=None,
    help="Temporary pitch applied only to the circuit-breaker above target.",
)
parser.add_argument(
    "--ik-lambda",
    type=float,
    default=None,
    help="Temporary DLS damping override for reachability diagnostics.",
)
parser.add_argument(
    "--probe-curobo-plan",
    action="store_true",
    help="At the below→above transition, test a cuRobo trajectory plan without executing it.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if "Franka" not in args_cli.task and "franka" not in args_cli.task:
    raise SystemExit(f"--task must be a Franka HiveBoard env, got {args_cli.task!r}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
import isaaclab_hiveboard.tasks  # noqa: F401


def _task_env_cfg(task: str):
    if "CircuitBreaker" in task or "Circuit-Breaker" in task:
        from isaaclab_hiveboard.tasks.franka.circuit_breaker.env import (
            FrankaCircuitBreakerEnvCfg,
        )

        return FrankaCircuitBreakerEnvCfg()
    if "LeverValve" in task or "Lever-Valve" in task or "Ball-Valve" in task:
        from isaaclab_hiveboard.tasks.franka.lever_valve.env import (
            FrankaLeverValveEnvCfg,
        )

        return FrankaLeverValveEnvCfg()
    raise SystemExit(f"Unknown Franka task: {task}")


def _target_frame_quat(roll_deg: float, pitch_deg: float) -> tuple[float, float, float, float]:
    """Return ``Ry(pitch) @ Rx(roll)`` in Isaac Lab's WXYZ convention."""
    roll = math.radians(roll_deg) / 2.0
    pitch = math.radians(pitch_deg) / 2.0
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    return (cp * cr, cp * sr, sp * cr, -sp * sr)


def _override_circuit_breaker_target_orientation(env_cfg) -> None:
    """Apply a diagnostic-only orientation to every circuit-breaker pose."""
    if args_cli.target_pitch_deg is None:
        return
    if not hasattr(env_cfg.scene, "target_frame"):
        raise SystemExit("--target-pitch-deg requires a scene with target_frame")
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

    quat = _target_frame_quat(args_cli.target_roll_deg, args_cli.target_pitch_deg)
    for frame in env_cfg.scene.target_frame.target_frames:
        frame.offset = OffsetCfg(pos=tuple(frame.offset.pos), rot=quat)
    print(
        "[INFO] diagnostic target override: "
        f"roll={args_cli.target_roll_deg:+.1f} deg "
        f"pitch={args_cli.target_pitch_deg:+.1f} deg"
    )


def _override_circuit_breaker_above_orientation(env_cfg) -> None:
    """Apply a diagnostic-only pitch to the final lever target alone."""
    if args_cli.above_pitch_deg is None:
        return
    for frame in env_cfg.scene.target_frame.target_frames:
        if frame.name != "lever_pivot_above":
            continue
        frame.offset = OffsetCfg(
            pos=tuple(frame.offset.pos),
            rot=_target_frame_quat(0.0, args_cli.above_pitch_deg),
        )
        print(
            "[INFO] diagnostic above-target pitch override: "
            f"{args_cli.above_pitch_deg:+.1f} deg"
        )
        return
    raise SystemExit("No lever_pivot_above target frame was found")


def _override_ik_damping(env_cfg) -> None:
    """Apply a recorder-only DLS damping value without altering task source."""
    if args_cli.ik_lambda is None:
        return
    controller = env_cfg.actions.arm_action.controller
    if controller.ik_method != "dls":
        raise SystemExit("--ik-lambda requires a DLS differential-IK controller")
    controller.ik_params = {"lambda_val": args_cli.ik_lambda}
    print(f"[INFO] diagnostic DLS damping override: lambda={args_cli.ik_lambda:g}")


def _route_pose_command(command: torch.Tensor, position_only: bool) -> torch.Tensor:
    if position_only:
        return command[:, 0:4]
    return command


def _probe_curobo_plan(base, cmd_term, arm_names, env_i: int) -> None:
    """Test planning the live TCP target through the package-local FR3 model."""
    from isaaclab_hiveboard.assets import ASSET_DIR
    from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
    from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp

    with curobo_compatible_warp():
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

        robot_cfg = load_curobo_robot_cfg(
            f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
            f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
        )
        planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=robot_cfg,
                self_collision_check=False,
                # This is a live single-arm probe alongside Isaac Sim. Keep
                # the planner footprint small; production can warm/capture it.
                use_cuda_graph=False,
                num_ik_seeds=4,
                num_trajopt_seeds=1,
                interpolation_dt=float(base.step_dt),
                interpolation_buffer_size=128,
                max_batch_size=1,
            )
        )
        try:
            env_ids = torch.tensor([env_i], device=base.device, dtype=torch.long)
            handler = cmd_term._command_handlers[3]
            target_pos, target_quat = handler.get_target_in_base_frame(env_ids)
            flange_pos, flange_quat = math_utils.subtract_frame_transforms(
                target_pos,
                target_quat,
                cmd_term._offset_pos[env_ids],
                cmd_term._offset_rot[env_ids],
            )
            current = JointState.from_position(
                base.scene["robot"].data.joint_pos[env_i, :7].unsqueeze(0),
                joint_names=arm_names,
            )
            goal = GoalToolPose.from_poses(
                {planner.tool_frames[0]: Pose(position=flange_pos, quaternion=flange_quat)},
                ordered_tool_frames=planner.tool_frames,
                num_goalset=1,
            )
            # The graph planner is useful for difficult obstacle scenes, but
            # for this one-arm, no-world probe it creates a large PRM query
            # with no benefit. Start with the bounded IK+TrajOpt attempt.
            result = planner.plan_pose(
                goal,
                current,
                max_attempts=1,
                enable_graph_attempt=99,
            )
            success = result is not None and bool(result.success.any().item())
            if success:
                plan = result.get_interpolated_plan()
                print(f"[INFO] cuRobo below→above plan: success, {plan.position.shape[-2]} waypoints")
            else:
                print("[INFO] cuRobo below→above plan: no feasible path")
        finally:
            planner.destroy()


def _ori_err_deg(q_a: torch.Tensor, q_b: torch.Tensor) -> float:
    mag = math_utils.quat_error_magnitude(q_a.unsqueeze(0), q_b.unsqueeze(0))[0]
    return float(torch.rad2deg(mag).item())


def _tcp_axes(
    quat: torch.Tensor,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    x = math_utils.quat_apply(quat.unsqueeze(0), quat.new_tensor([[1.0, 0.0, 0.0]]))[0]
    z = math_utils.quat_apply(quat.unsqueeze(0), quat.new_tensor([[0.0, 0.0, 1.0]]))[0]
    return (
        (float(x[0]), float(x[1]), float(x[2])),
        (float(z[0]), float(z[1]), float(z[2])),
    )


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine alignment for unit frame axes, with 1/-1 meaning aligned/flipped."""
    return float(torch.sum(a * b).item())


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    env_cfg = _task_env_cfg(args_cli.task)
    _override_circuit_breaker_target_orientation(env_cfg)
    _override_circuit_breaker_above_orientation(env_cfg)
    _override_ik_damping(env_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.position_only:
        env_cfg.actions.arm_action.controller.command_type = "position"
        print("[INFO] IK ablation: position-only.")

    env = gym.make(id=args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    step_dt = float(base.step_dt)
    episode_steps = int(getattr(base, "max_episode_length", 0) or 0)
    if episode_steps <= 0:
        episode_steps = max(1, math.ceil(float(base.cfg.episode_length_s) / step_dt))
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else episode_steps

    robot = base.scene["robot"]
    cmd_term = base.command_manager.get_term("pose_command")
    cmd_names = []
    for cfg in cmd_term.cfg.commands:
        name = type(cfg).__name__.removesuffix("Cfg")
        frame = getattr(cfg, "target_frame_name", None)
        cmd_names.append(f"{name}:{frame}" if frame else name)
    arm_ids, arm_names = robot.find_joints(
        base.cfg.actions.arm_action.joint_names
    )
    if not isinstance(arm_ids, torch.Tensor):
        arm_ids = torch.as_tensor(arm_ids, device=base.device, dtype=torch.long)

    breaker = None
    breaker_j = None
    if "circuit_breaker" in base.scene.keys():
        breaker = base.scene["circuit_breaker"]
        jids, _ = breaker.find_joints("RevoluteJoint")
        if jids:
            breaker_j = jids[0]

    tag = os.path.basename(args_cli.task).replace("/", "_")
    if args_cli.position_only:
        tag += "_posonly"
    out_dir = args_cli.out_dir or os.path.join(
        "logs", "franka_debug", f"{tag}_{_stamp()}"
    )
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict] = []
    obs, _ = env.reset()
    env_i = 0
    prev_q = robot.data.joint_pos[env_i, arm_ids].clone()
    root0 = robot.data.root_pos_w[env_i].clone()
    quat0 = robot.data.root_quat_w[env_i].clone()
    curobo_plan_probed = False

    print(
        f"[INFO] {args_cli.task}  steps<={max_steps}  dt={step_dt:.4f}s  "
        f"fix_root={getattr(robot.cfg.spawn.articulation_props, 'fix_root_link', None)}  "
        f"out={out_dir}"
    )

    for step in range(max_steps):
        with torch.inference_mode():
            command = obs["policy"]["command"] if isinstance(obs, dict) else None
            if command is None:
                action = torch.zeros(env.action_space.shape, device=base.device)
            else:
                action = _route_pose_command(command, args_cli.position_only)
            obs, _, terminated, truncated, _ = env.step(action)

            env_ids = torch.tensor([env_i], device=base.device, dtype=torch.long)
            ee_pos_b, ee_quat_b = cmd_term._get_ee_in_base_frame(env_ids)
            cmd = cmd_term.command[env_i]
            cmd_pos = cmd[1:4]
            cmd_quat = cmd[4:8]
            cmd_idx = int(cmd_term._current_command_idx[env_i].item())
            cmd_name = cmd_names[cmd_idx] if cmd_idx < len(cmd_names) else "done"
            if args_cli.probe_curobo_plan and not curobo_plan_probed and cmd_idx == 3:
                _probe_curobo_plan(base, cmd_term, arm_names, env_i)
                curobo_plan_probed = True
            pos_err = float(torch.linalg.vector_norm(ee_pos_b[0] - cmd_pos).item())
            ori_err = (
                0.0
                if args_cli.position_only
                else _ori_err_deg(ee_quat_b[0], cmd_quat)
            )
            q = robot.data.joint_pos[env_i, arm_ids]
            dq = q - prev_q
            prev_q = q.clone()
            qvel = robot.data.joint_vel[env_i, arm_ids]
            root = robot.data.root_pos_w[env_i]
            root_quat = robot.data.root_quat_w[env_i]
            root_drift = float(torch.linalg.vector_norm(root - root0).item())
            root_tilt = _ori_err_deg(root_quat, quat0)
            tcp_x, tcp_z = _tcp_axes(ee_quat_b[0])
            cmd_x_ax, cmd_z_ax = _tcp_axes(cmd_quat)

            # Trace the complete orientation chain.  A GoToFrame handler owns
            # a raw scene target; ``cmd`` is its interpolated TCP command; the
            # differential-IK action converts that TCP command into a flange
            # target using the same fixed body offset used for measured TCP.
            handler = (
                cmd_term._command_handlers[cmd_idx]
                if cmd_idx < len(cmd_term._command_handlers)
                else None
            )
            frame = getattr(handler, "_frame", None)
            frame_idx = getattr(handler, "_frame_idx", None)
            has_target_frame = frame is not None and frame_idx is not None
            if has_target_frame:
                frame_pos_w = frame.data.target_pos_w[env_i, frame_idx]
                frame_quat_w = frame.data.target_quat_w[env_i, frame_idx]
                frame_pos_b, frame_quat_b = math_utils.subtract_frame_transforms(
                    root.unsqueeze(0),
                    root_quat.unsqueeze(0),
                    frame_pos_w.unsqueeze(0),
                    frame_quat_w.unsqueeze(0),
                )
                frame_pos_b, frame_quat_b = frame_pos_b[0], frame_quat_b[0]
            else:
                frame_pos_b = torch.full_like(cmd_pos, float("nan"))
                frame_quat_b = torch.full_like(cmd_quat, float("nan"))

            flange_pos_b = robot.data.body_pos_w[env_i, cmd_term._body_idx]
            flange_quat_b = robot.data.body_quat_w[env_i, cmd_term._body_idx]
            flange_pos_b, flange_quat_b = math_utils.subtract_frame_transforms(
                root.unsqueeze(0),
                root_quat.unsqueeze(0),
                flange_pos_b.unsqueeze(0),
                flange_quat_b.unsqueeze(0),
            )
            flange_pos_b, flange_quat_b = flange_pos_b[0], flange_quat_b[0]
            if cmd_term._offset_pos is not None and cmd_term._offset_rot is not None:
                flange_cmd_pos_b, flange_cmd_quat_b = math_utils.subtract_frame_transforms(
                    cmd_pos.unsqueeze(0),
                    cmd_quat.unsqueeze(0),
                    cmd_term._offset_pos[env_i].unsqueeze(0),
                    cmd_term._offset_rot[env_i].unsqueeze(0),
                )
                flange_cmd_pos_b, flange_cmd_quat_b = (
                    flange_cmd_pos_b[0],
                    flange_cmd_quat_b[0],
                )
            else:
                flange_cmd_pos_b, flange_cmd_quat_b = cmd_pos, cmd_quat

            flange_x, flange_z = _tcp_axes(flange_quat_b)
            flange_cmd_x, flange_cmd_z = _tcp_axes(flange_cmd_quat_b)
            cmd_x_t = torch.tensor(cmd_x_ax, device=base.device)
            cmd_z_t = torch.tensor(cmd_z_ax, device=base.device)
            tcp_x_t = torch.tensor(tcp_x, device=base.device)
            tcp_z_t = torch.tensor(tcp_z, device=base.device)
            if has_target_frame:
                frame_x, frame_z = _tcp_axes(frame_quat_b)
                frame_x_t = torch.tensor(frame_x, device=base.device)
                frame_z_t = torch.tensor(frame_z, device=base.device)
                frame_cmd_x_dot = _dot(frame_x_t, cmd_x_t)
                frame_cmd_z_dot = _dot(frame_z_t, cmd_z_t)
                frame_tcp_x_dot = _dot(frame_x_t, tcp_x_t)
                frame_tcp_z_dot = _dot(frame_z_t, tcp_z_t)
            else:
                frame_x = frame_z = (float("nan"),) * 3
                frame_cmd_x_dot = frame_cmd_z_dot = float("nan")
                frame_tcp_x_dot = frame_tcp_z_dot = float("nan")
            lever_rad = (
                float(breaker.data.joint_pos[env_i, breaker_j].item())
                if breaker is not None and breaker_j is not None
                else float("nan")
            )

            row = {
                "step": step,
                "time_s": step * step_dt,
                "command_idx": cmd_idx,
                "command_name": cmd_name,
                "gripper": float(cmd[0].item()),
                "pos_err_m": pos_err,
                "ori_err_deg": ori_err,
                "root_drift_m": root_drift,
                "root_tilt_deg": root_tilt,
                "max_abs_dq": float(dq.abs().max().item()),
                "max_abs_qvel": float(qvel.abs().max().item()),
                "ee_x_b": float(ee_pos_b[0, 0]),
                "ee_y_b": float(ee_pos_b[0, 1]),
                "ee_z_b": float(ee_pos_b[0, 2]),
                "ee_qw": float(ee_quat_b[0, 0]),
                "ee_qx": float(ee_quat_b[0, 1]),
                "ee_qy": float(ee_quat_b[0, 2]),
                "ee_qz": float(ee_quat_b[0, 3]),
                "cmd_x_b": float(cmd_pos[0]),
                "cmd_y_b": float(cmd_pos[1]),
                "cmd_z_b": float(cmd_pos[2]),
                "cmd_qw": float(cmd_quat[0]),
                "cmd_qx": float(cmd_quat[1]),
                "cmd_qy": float(cmd_quat[2]),
                "cmd_qz": float(cmd_quat[3]),
                "root_x": float(root[0]),
                "root_y": float(root[1]),
                "root_z": float(root[2]),
                "tcp_x_x": tcp_x[0],
                "tcp_x_y": tcp_x[1],
                "tcp_x_z": tcp_x[2],
                "tcp_z_x": tcp_z[0],
                "tcp_z_y": tcp_z[1],
                "tcp_z_z": tcp_z[2],
                "cmd_tcp_x_z": cmd_x_ax[2],
                "cmd_tcp_z_x": cmd_z_ax[0],
                "frame_x_b_x": frame_x[0],
                "frame_x_b_y": frame_x[1],
                "frame_x_b_z": frame_x[2],
                "frame_z_b_x": frame_z[0],
                "frame_z_b_y": frame_z[1],
                "frame_z_b_z": frame_z[2],
                "frame_cmd_x_dot": frame_cmd_x_dot,
                "frame_cmd_z_dot": frame_cmd_z_dot,
                "frame_tcp_x_dot": frame_tcp_x_dot,
                "frame_tcp_z_dot": frame_tcp_z_dot,
                "cmd_tcp_x_dot": _dot(cmd_x_t, tcp_x_t),
                "cmd_tcp_z_dot": _dot(cmd_z_t, tcp_z_t),
                "flange_x_b_x": flange_x[0],
                "flange_x_b_y": flange_x[1],
                "flange_x_b_z": flange_x[2],
                "flange_z_b_x": flange_z[0],
                "flange_z_b_y": flange_z[1],
                "flange_z_b_z": flange_z[2],
                "flange_cmd_x_b_x": flange_cmd_x[0],
                "flange_cmd_x_b_y": flange_cmd_x[1],
                "flange_cmd_x_b_z": flange_cmd_x[2],
                "flange_cmd_z_b_x": flange_cmd_z[0],
                "flange_cmd_z_b_y": flange_cmd_z[1],
                "flange_cmd_z_b_z": flange_cmd_z[2],
                "lever_rad": lever_rad,
            }
            for name, value in zip(arm_names, q.tolist()):
                row[f"q_{name}"] = float(value)
            rows.append(row)

            if args_cli.log_every and step % args_cli.log_every == 0:
                alignment = (
                    f"frame\u2192cmd X/Z={frame_cmd_x_dot:+.2f}/{frame_cmd_z_dot:+.2f}  "
                    f"cmd\u2192tcp X/Z={row['cmd_tcp_x_dot']:+.2f}/{row['cmd_tcp_z_dot']:+.2f}"
                    if has_target_frame
                    else "no target frame"
                )
                print(
                    f"  step={step:4d} {cmd_name:16s}  "
                    f"pos_err={pos_err:.3f}m  ori_err={ori_err:6.1f}deg  "
                    f"|dq|={row['max_abs_dq']:.3f}  "
                    f"|qvel|={row['max_abs_qvel']:.2f}  "
                    f"ee_z={row['ee_z_b']:+.3f} cmd_z={row['cmd_z_b']:+.3f}  "
                    f"lever={lever_rad:+.3f}  {alignment}"
                )

            term = terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)
            trunc = truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
            if term or trunc:
                print(f"[INFO] episode end at step {step}  terminated={term} truncated={trunc}")
                break

    env.close()

    csv_path = os.path.join(out_dir, "traj.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pos = np.array([r["pos_err_m"] for r in rows])
    ori = np.array([r["ori_err_deg"] for r in rows])
    dq = np.array([r["max_abs_dq"] for r in rows])
    qvel = np.array([r["max_abs_qvel"] for r in rows])
    drift = np.array([r["root_drift_m"] for r in rows])
    tilt = np.array([r["root_tilt_deg"] for r in rows])
    skip = min(10, max(1, len(rows) // 10))
    # Ignore the first sample (dq vs pre-reset) and the terminal/reset
    # sample, which writes default joints in one shot.
    active = [
        i
        for i, row in enumerate(rows)
        if row["command_name"] != "done" and 0 < i < len(rows) - 1
    ]
    if not active:
        active = list(range(len(rows)))
    dq_active = dq[active]
    qvel_active = qvel[active]
    pos_active = pos[active]
    ori_active = ori[active]

    checks = {
        "root_fixed": bool(drift.max() < 0.01 and tilt.max() < 2.0),
        "no_joint_explosion": bool(dq_active.max() < 0.5),
        "joint_vel_bounded": bool(qvel_active.max() < 25.0),
        "tracking_improves": bool(
            np.median(pos[: max(skip, 1)]) > np.median(pos[-max(skip, 1) :])
            or pos_active[-1] < 0.05
        ),
        "final_pos_err_lt_5cm": bool(pos_active[-1] < 0.05),
        "final_ori_err_lt_25deg": bool(
            args_cli.position_only or ori_active[-1] < 25.0
        ),
        "no_nan": bool(np.isfinite(pos).all() and np.isfinite(ori).all()),
    }
    summary = {
        "task": args_cli.task,
        "position_only": bool(args_cli.position_only),
        "n_steps": len(rows),
        "last_command": rows[-1]["command_name"],
        "last_command_idx": rows[-1]["command_idx"],
        "max_root_drift_m": float(drift.max()),
        "max_root_tilt_deg": float(tilt.max()),
        "max_abs_dq_rad": float(dq_active.max()),
        "max_abs_qvel": float(qvel_active.max()),
        "pos_err_m": {
            "first": float(pos[0]),
            "median": float(np.median(pos)),
            "last": float(pos[-1]),
            "max": float(pos.max()),
        },
        "ori_err_deg": {
            "first": float(ori[0]),
            "median": float(np.median(ori)),
            "last": float(ori[-1]),
            "max": float(ori.max()),
        },
        "checks": checks,
        "pass": all(checks.values()),
        "csv": os.path.abspath(csv_path),
        "phases": [],
    }
    phase_rows = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["command_idx"] != rows[start]["command_idx"]:
            chunk = rows[start:i]
            dz_cmd = chunk[-1]["cmd_z_b"] - chunk[0]["cmd_z_b"]
            dz_ee = chunk[-1]["ee_z_b"] - chunk[0]["ee_z_b"]
            phase = {
                "idx": chunk[0]["command_idx"],
                "name": chunk[0]["command_name"],
                "n": len(chunk),
                "d_cmd_z_m": dz_cmd,
                "d_ee_z_m": dz_ee,
                "max_ori_err_deg": max(c["ori_err_deg"] for c in chunk),
                "lever_start_rad": chunk[0]["lever_rad"],
                "lever_end_rad": chunk[-1]["lever_rad"],
            }
            phase_rows.append(phase)
            start = i
    summary["phases"] = phase_rows
    if "CircuitBreaker" in args_cli.task or "Circuit-Breaker" in args_cli.task:
        up_phases = [p for p in phase_rows if p["idx"] == 3]
        checks["down_to_up_cmd_z"] = bool(
            up_phases and up_phases[0]["d_cmd_z_m"] >= 0.10
        )
        checks["down_to_up_ee_z"] = bool(
            up_phases and up_phases[0]["d_ee_z_m"] >= 0.07
        )
    summary["checks"] = checks
    summary["pass"] = all(checks.values())
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    txt_path = os.path.join(out_dir, "summary.txt")
    lines = [
        f"task: {summary['task']}",
        f"position_only: {summary['position_only']}",
        f"steps: {summary['n_steps']}  last_cmd: {summary['last_command']}",
        f"root drift: {summary['max_root_drift_m']:.4f} m   tilt: {summary['max_root_tilt_deg']:.2f} deg",
        f"max |dq|: {summary['max_abs_dq_rad']:.3f} rad/step   max |qvel|: {summary['max_abs_qvel']:.2f} rad/s",
        f"pos_err m  first={pos[0]:.4f}  last={pos[-1]:.4f}  max={pos.max():.4f}",
        f"ori_err deg first={ori[0]:.1f}  last={ori[-1]:.1f}  max={ori.max():.1f}",
        "",
        "checks:",
    ]
    for name, ok in checks.items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if phase_rows:
        lines.append("")
        lines.append("phases: idx name n d_cmd_z d_ee_z max_ori lever0->lever1")
        for p in phase_rows:
            lines.append(
                f"  {p['idx']} {p['name']:16s} n={p['n']:3d}  "
                f"d_cmd_z={p['d_cmd_z_m']:+.3f} d_ee_z={p['d_ee_z_m']:+.3f}  "
                f"ori_max={p['max_ori_err_deg']:.1f}  "
                f"lever={p['lever_start_rad']:+.3f}->{p['lever_end_rad']:+.3f}"
            )
    lines.append("")
    lines.append("OVERALL: " + ("PASS" if summary["pass"] else "FAIL"))
    lines.append(f"csv: {csv_path}")
    text = "\n".join(lines) + "\n"
    with open(txt_path, "w") as handle:
        handle.write(text)
    print()
    print(text)
    print(f"[INFO] wrote {json_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
