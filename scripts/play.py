#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless HiveBoard task player (Newton / Isaac Lab 3)."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping
from datetime import datetime

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_hiveboard  # noqa: F401
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab_hiveboard.mdp.recorders import SpotManipulationRecorderCfg
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


DEFAULT_TASK = "Isaac-HiveBoard-Spot-BallValve-Play-v0"
CONTACT_SENSOR_NAMES = (
    "finger_contact",
    "jaw_contact",
    "wr1_contact",
    "wr0_contact",
    "el1_contact",
    "el0_contact",
)


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _all_finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    return True


def _route_command(env, command: torch.Tensor) -> torch.Tensor:
    terms = list(env.action_manager.active_terms)
    dims = list(env.action_manager.action_term_dim)
    if terms == ["gripper_action", "arm_action"] and dims == [1, 7]:
        return command
    if terms == ["gripper_action", "arm_action"] and dims == [1, 3]:
        return command[:, 0:4]
    if terms == ["arm_action", "gripper_action"] and dims == [7, 1]:
        return torch.cat((command[:, 1:8], command[:, 0:1]), dim=-1)
    if terms == ["arm_action", "gripper_action"] and dims == [3, 1]:
        return torch.cat((command[:, 1:4], command[:, 0:1]), dim=-1)
    if terms == ["arm_action"] and dims == [7]:
        return command
    return command


def _apply_collision_only(base) -> bool:
    """Hide visual geometry so Newton viewers show only collision shapes."""
    switched = False
    for viz in base.sim.visualizers:
        viewer = getattr(viz, "_viewer", None)
        if viewer is None or not hasattr(viewer, "show_visual"):
            continue
        viewer.show_visual = False
        viewer.show_collision = True
        switched = True
    return switched


def _visualizers_alive(sim) -> bool:
    if not getattr(sim, "visualizers", None):
        return False
    return any(viz.is_running() and not viz.is_closed for viz in sim.visualizers)


def _force_norm(forces) -> float:
    if forces is None:
        return float("nan")
    tensor = _as_torch(forces)
    if tensor is None:
        return float("nan")
    return float(torch.linalg.vector_norm(tensor.reshape(-1, 3), dim=-1).max().item())


def _print_contact(base, step: int, env_index: int) -> None:
    debug_env = min(env_index, base.num_envs - 1)
    report = {}
    for name in CONTACT_SENSOR_NAMES:
        if name not in base.scene.keys():
            continue
        sensor = base.scene[name]
        net = _as_torch(sensor.data.net_forces_w)
        matrix = sensor.data.force_matrix_w
        report[name] = {
            "net_N": _force_norm(None if net is None else net[debug_env]),
            "valve_N": _force_norm(None if matrix is None else _as_torch(matrix)[debug_env]),
        }
    if report:
        print(f"[CONTACT] step={step} {report}", flush=True)


def _print_pose(base, step: int, env_index: int) -> None:
    debug_env = min(env_index, base.num_envs - 1)
    if "joint_command" in base.command_manager.active_terms and "ball_valve" in base.scene.keys():
        valve = base.scene["ball_valve"]
        joint_pos = _as_torch(valve.data.joint_pos)[debug_env].detach().cpu().tolist()
        command = _as_torch(base.command_manager.get_term("joint_command").command)[debug_env]
        print(
            f"[POSE] step={step} valve_joint={joint_pos} "
            f"joint_cmd={command.detach().cpu().tolist()}",
            flush=True,
        )
        return
    if "pose_command" not in base.command_manager.active_terms:
        return
    command_term = base.command_manager.get_term("pose_command")
    command_idx = getattr(command_term, "_current_command_idx", None)
    tracked_name = getattr(base.cfg.viewer, "asset_name", None)
    tracked = base.scene[tracked_name] if tracked_name in base.scene.keys() else None
    joint_pos = None
    if tracked is not None and hasattr(tracked.data, "joint_pos"):
        joint_pos = _as_torch(tracked.data.joint_pos)[debug_env].detach().cpu().tolist()
    command_text = None
    if command_idx is not None:
        command_text = _as_torch(command_idx).detach().cpu().tolist()
    print(
        f"[POSE] step={step} command_idx={command_text} joint_pos={joint_pos}",
        flush=True,
    )
    if command_idx is None:
        return
    active_idx = int(_as_torch(command_idx)[debug_env].item())
    handlers = getattr(command_term, "_command_handlers", ())
    if active_idx >= len(handlers):
        return
    env_ids = torch.tensor([debug_env], device=base.device, dtype=torch.long)
    ee_pos_b, ee_quat_b = command_term._get_ee_in_base_frame(env_ids)
    target_pos_b, target_quat_b = handlers[active_idx].get_target_in_base_frame(env_ids)
    ee_pos_b = _as_torch(ee_pos_b)
    ee_quat_b = _as_torch(ee_quat_b)
    target_pos_b = _as_torch(target_pos_b)
    target_quat_b = _as_torch(target_quat_b)
    pos_err = torch.linalg.vector_norm(ee_pos_b - target_pos_b, dim=-1)
    ori_err = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)
    print(
        f"[POSE] env={debug_env} ee_pos_b={ee_pos_b.cpu().tolist()} "
        f"target_pos_b={target_pos_b.cpu().tolist()} "
        f"pos_err={pos_err.cpu().tolist()} "
        f"ori_err_deg={torch.rad2deg(ori_err).cpu().tolist()}",
        flush=True,
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK, help="Gym task id.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
    parser.add_argument("--seed", type=int, default=None, help="Environment reset seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many environment steps. Default: run until the visualizer closes, or one episode if headless.",
    )
    parser.add_argument(
        "--pose-debug",
        action="store_true",
        default=False,
        help="Print EE tracking error and (when present) contact forces.",
    )
    parser.add_argument(
        "--pose-debug-interval",
        type=int,
        default=30,
        help="Print a compact pose/contact report every N environment steps (default: 30).",
    )
    parser.add_argument(
        "--pose-debug-env",
        type=int,
        default=0,
        help="Environment index to inspect with --pose-debug / --contact-debug (default: 0).",
    )
    parser.add_argument(
        "--contact-debug",
        action="store_true",
        default=False,
        help="Print finger/jaw net and valve-filtered contact force norms.",
    )
    parser.add_argument(
        "--ee-debug",
        action="store_true",
        default=False,
        help="Print gripper-body vs offset-TCP axes once after reset.",
    )
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="Newton visualizer shows only collision geometry. No-op with --visualizer none.",
    )
    parser.add_argument(
        "--joint-log",
        default=None,
        help="Directory for commanded vs measured joint CSV/plots. "
        "Default: logs/joint_tracking/<timestamp> when the task has joint_command.",
    )
    parser.add_argument(
        "--no-joint-log",
        action="store_true",
        help="Disable joint tracking logs even for joint-trajectory tasks.",
    )
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    if args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        args.visualizer = ["newton"]
    return args, hydra_args


def main() -> int:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    if args.pose_debug:
        if hasattr(env_cfg.scene, "target_frame"):
            env_cfg.scene.target_frame.debug_vis = True
        if hasattr(env_cfg.scene, "ee_frame"):
            env_cfg.scene.ee_frame.debug_vis = True

    if args.contact_debug:
        missing = [name for name in CONTACT_SENSOR_NAMES if not hasattr(env_cfg.scene, name)]
        if missing:
            raise ValueError("--contact-debug is only supported by tasks with gripper contact sensors")

    if args.collision_only:
        try:
            from isaaclab_visualizers.newton import NewtonVisualizerCfg
        except ImportError as err:
            raise SystemExit(
                "--collision-only needs the Newton visualizer backend "
                "(pip install isaaclab_visualizers[newton])."
            ) from err
        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(show_collision=True)]

    # Hydra round-trips class types to strings. Rebuild a live recorder cfg so
    # HDF5DatasetFileHandler and term class_type stay callable.
    if getattr(env_cfg, "recorders", None) is not None:
        src = env_cfg.recorders
        rec = SpotManipulationRecorderCfg()
        rec.dataset_export_dir_path = getattr(
            src, "dataset_export_dir_path", rec.dataset_export_dir_path
        )
        rec.dataset_filename = getattr(src, "dataset_filename", rec.dataset_filename)
        rec.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        rec.export_in_close = True
        env_cfg.recorders = rec
        print(
            "[INFO] Recording episodes to "
            f"{rec.dataset_export_dir_path}/{rec.dataset_filename}.hdf5"
        )

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        if args.collision_only and not _apply_collision_only(base):
            print(
                "[WARN] --collision-only: no Newton viewer active "
                "(use --visualizer newton, not none).",
                file=sys.stderr,
            )

        step_dt = float(base.cfg.sim.dt) * float(base.cfg.decimation)
        episode_steps = int(getattr(base, "max_episode_length", 0) or 0)
        if episode_steps <= 0:
            episode_steps = max(1, math.ceil(float(base.cfg.episode_length_s) / step_dt))

        reset_kwargs = {} if args.seed is None else {"seed": args.seed}
        obs, _ = env.reset(**reset_kwargs)

        if args.ee_debug:
            from isaaclab_hiveboard.assets.end_effector import SPOT_EE, print_ee_offset_report

            print_ee_offset_report(base, SPOT_EE)

        joint_log = None
        want_joint_log = not args.no_joint_log and (
            args.joint_log is not None or "joint_command" in base.command_manager.active_terms
        )
        if want_joint_log:
            from isaaclab_hiveboard.utils.joint_traj import JointTrajDumper

            out_dir = args.joint_log or os.path.join(
                "logs",
                "joint_tracking",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            joint_log = JointTrajDumper(base, out_dir, env_index=args.pose_debug_env)
            print(f"[INFO] Joint tracking log: {out_dir}")

        count = 0
        try:
            while True:
                if base.sim.visualizers and not _visualizers_alive(base.sim):
                    break
                if args.max_steps is not None and count >= args.max_steps:
                    break
                if not base.sim.visualizers and args.max_steps is None and count >= episode_steps:
                    break

                if not _all_finite(obs):
                    raise FloatingPointError(f"Non-finite observation at step {count}")

                if (
                    isinstance(obs, dict)
                    and isinstance(obs.get("policy"), dict)
                    and "command" in obs["policy"]
                ):
                    action = _route_command(base, obs["policy"]["command"])
                else:
                    action = torch.zeros(env.action_space.shape, device=base.device)
                if not _all_finite(action):
                    raise FloatingPointError(f"Non-finite action at step {count}")

                with torch.inference_mode():
                    obs, _, terminated, truncated, _ = env.step(action)
                count += 1
                if joint_log is not None:
                    joint_log.sample(count, action)

                log_now = (args.pose_debug or args.contact_debug) and (
                    count % max(args.pose_debug_interval, 1) == 0
                )
                if log_now and args.contact_debug:
                    _print_contact(base, count, args.pose_debug_env)
                if log_now and args.pose_debug:
                    _print_pose(base, count, args.pose_debug_env)

                term = terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)
                trunc = truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
                if term or trunc:
                    if _visualizers_alive(base.sim) and args.max_steps is None:
                        obs, _ = env.reset()
                        continue
                    break
        finally:
            if joint_log is not None and joint_log._rows:
                joint_log.save()
            env.close()

    print(f"[INFO] Played {count} steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
