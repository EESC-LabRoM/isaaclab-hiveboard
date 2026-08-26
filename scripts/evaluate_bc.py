#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an MLP BC checkpoint in the newer Spot lever-valve PLAY environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Evaluate a BC checkpoint on the Spot lever-valve PLAY environment."
)
parser.add_argument("--policy_path", type=Path, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument(
    "--task_direction",
    choices=("open", "close", "both"),
    default="open",
    help="Valve task direction to evaluate (default: open).",
)
parser.add_argument(
    "--success_threshold_deg",
    type=float,
    default=15.0,
    help="Maximum final valve-goal error counted as success.",
)
parser.add_argument("--max_steps", type=int, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=600)
parser.add_argument("--video_folder", type=Path, default=Path("videos/run_bc"))
parser.add_argument(
    "--metrics_dir", type=Path, default=Path("logs/bc_evaluations")
)
parser.add_argument("--evaluation_name", type=str, default=None)
parser.add_argument("--real-time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.policy_path is None:
    parser.error("--policy_path is required")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be greater than zero")
if args_cli.success_threshold_deg <= 0.0:
    parser.error("--success_threshold_deg must be greater than zero")
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be greater than zero")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac Sim and project imports must follow AppLauncher construction."""

import csv
import json
import math
import os
import time
from datetime import datetime
from typing import Any

import gymnasium as gym
import torch
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.utils.dict import print_dict
from tqdm import tqdm

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.terminations import (
    DeltaCollectionTerminationsCfg,
)
from isaaclab_hiveboard.tasks.spot.ball_valve.env import SpotBallValveEnvCfg_PLAY

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy.bc.policy import BehaviorCloningPolicy


TASK_ID = "Isaac-HiveBoard-Spot-BallValve-Play-v0"
OBSERVATION_GROUP = "diffusion_policy"


def policy_observation(observations: dict[str, Any]) -> torch.Tensor:
    """Extract and validate the flat policy observation group."""
    observation = observations.get(OBSERVATION_GROUP)
    if not isinstance(observation, torch.Tensor) or observation.ndim != 2:
        shape = tuple(observation.shape) if isinstance(observation, torch.Tensor) else None
        raise RuntimeError(
            f"Expected tensor observation group {OBSERVATION_GROUP!r}, got {shape}"
        )
    return observation


def evaluation_observation(
    observations: dict[str, Any], num_envs: int
) -> dict[str, torch.Tensor]:
    """Extract physical signals used in per-episode metrics."""
    evaluation = observations.get("evaluation")
    required = {
        "valve_angle": 1,
        "valve_current_angle": 1,
        "valve_goal_angle": 1,
        "valve_task_direction": 1,
    }
    if not isinstance(evaluation, dict):
        raise RuntimeError("Environment did not return dictionary group 'evaluation'")
    for name, dimension in required.items():
        value = evaluation.get(name)
        expected = (num_envs, dimension)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise RuntimeError(
                f"Evaluation observation {name!r} has shape {shape}, expected {expected}"
            )
    return evaluation


def sanitize_absolute_actions(actions: torch.Tensor) -> torch.Tensor:
    """Bound the binary gripper and normalize the absolute TCP quaternion."""
    if actions.ndim != 2 or actions.shape[-1] != 8:
        raise RuntimeError(f"Expected absolute actions shaped (N, 8), got {tuple(actions.shape)}")
    sanitized = actions.clone()
    sanitized[:, 0].clamp_(-1.0, 1.0)
    quaternion = sanitized[:, 4:8]
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    valid = norm[:, 0] > 1.0e-6
    quaternion[valid] /= norm[valid]
    quaternion[~valid] = quaternion.new_tensor((1.0, 0.0, 0.0, 0.0))
    return sanitized


def write_metrics(
    output_dir: Path, records: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[Path, Path]:
    """Write episode CSV and aggregate JSON metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.json"
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        if records:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    successes = sum(bool(record["success"]) for record in records)
    count = len(records)
    payload = {
        "metadata": metadata,
        "aggregate": {
            "episodes": count,
            "successes": successes,
            "failures": count - successes,
            "success_rate": successes / count if count else 0.0,
            "mean_duration_s": (
                sum(float(record["duration_s"]) for record in records) / count
                if count
                else 0.0
            ),
            "mean_pre_terminal_goal_error_deg": (
                sum(float(record["pre_terminal_goal_error_deg"]) for record in records)
                / count
                if count
                else 0.0
            ),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return episodes_path, summary_path


def main() -> None:
    """Run closed-loop BC evaluation and save metrics and rollout datasets."""
    policy_path = args_cli.policy_path.resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {policy_path}")

    evaluation_name = args_cli.evaluation_name or datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    output_dir = (args_cli.metrics_dir / evaluation_name).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_dir}")
    output_dir.mkdir(parents=True)

    env_cfg = SpotBallValveEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.commands.pose_command.open_task_prob = {
        "open": 1.0,
        "close": 0.0,
        "both": 0.5,
    }[args_cli.task_direction]
    env_cfg.terminations = DeltaCollectionTerminationsCfg()
    env_cfg.terminations.success.params["threshold_rad"] = math.radians(
        args_cli.success_threshold_deg
    )
    env_cfg.recorders.dataset_export_dir_path = str(output_dir)
    env_cfg.recorders.dataset_filename = "rollouts"
    env_cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    )

    env = gym.make(
        TASK_ID,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.abspath(args_cli.video_folder),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "fps": int(round(1.0 / env.unwrapped.step_dt)),
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    action_terms = list(base_env.action_manager.active_terms)
    action_dims = list(base_env.action_manager.action_term_dim)
    if action_terms != ["gripper_action", "arm_action"] or action_dims != [1, 7]:
        raise RuntimeError(
            "Expected newer absolute action layout gripper(1), arm pose(7); "
            f"received terms={action_terms}, dims={action_dims}"
        )

    policy = BehaviorCloningPolicy.load(policy_path, device=base_env.device)
    observations, _ = env.reset()
    observation = policy_observation(observations)
    evaluation = evaluation_observation(observations, args_cli.num_envs)
    if observation.shape[-1] != policy.obs_dim or policy.action_dim != 8:
        raise RuntimeError(
            f"Checkpoint/environment mismatch: observation={tuple(observation.shape)}, "
            f"policy obs_dim={policy.obs_dim}, action_dim={policy.action_dim}"
        )
    if not torch.isfinite(observation).all():
        raise RuntimeError("Initial policy observation contains NaN or Inf")

    history = observation.unsqueeze(1).repeat(1, policy.n_obs_steps, 1)
    episodes_per_env, remainder = divmod(args_cli.num_episodes, args_cli.num_envs)
    targets = torch.full(
        (args_cli.num_envs,), episodes_per_env, dtype=torch.long, device=base_env.device
    )
    targets[:remainder] += 1
    counts = torch.zeros_like(targets)
    episode_steps = torch.zeros_like(targets)
    records: list[dict[str, Any]] = []
    valve_span_deg = math.degrees(
        abs(
            env_cfg.commands.pose_command.valve_joint_open
            - env_cfg.commands.pose_command.valve_joint_closed
        )
    )
    total_steps = 0
    progress = tqdm(total=args_cli.num_episodes, desc="Evaluating BC", unit="episodes")

    print(
        "[INFO] Evaluation contract: "
        f"env={TASK_ID}, direction={args_cli.task_direction}, "
        f"observations={policy.obs_dim}, actions={policy.action_dim}, "
        f"threshold={args_cli.success_threshold_deg:.1f} deg"
    )

    while simulation_app.is_running() and len(records) < args_cli.num_episodes:
        start_time = time.time()
        with torch.inference_mode():
            raw_actions = policy.predict_action({"obs": history})["action"][:, 0]
            if not torch.isfinite(raw_actions).all():
                raise RuntimeError(f"Policy produced NaN or Inf at step {total_steps}")
            actions = sanitize_absolute_actions(raw_actions)
            pre_terminal_evaluation = evaluation
            observations, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            episode_steps += 1

            include = done & (counts < targets)
            for env_id in include.nonzero(as_tuple=False).flatten().tolist():
                current = float(
                    pre_terminal_evaluation["valve_current_angle"][env_id, 0].item()
                )
                goal = float(
                    pre_terminal_evaluation["valve_goal_angle"][env_id, 0].item()
                )
                goal_error_normalized = abs(current - goal)
                goal_error_deg = goal_error_normalized * valve_span_deg
                records.append(
                    {
                        "episode": len(records),
                        "env_id": env_id,
                        "success": bool(terminated[env_id].item()),
                        "time_out": bool(truncated[env_id].item()),
                        "goal_within_threshold": goal_error_deg
                        <= args_cli.success_threshold_deg,
                        "steps": int(episode_steps[env_id].item()),
                        "duration_s": float(episode_steps[env_id].item()) * base_env.step_dt,
                        "task_direction": float(
                            pre_terminal_evaluation["valve_task_direction"][env_id, 0].item()
                        ),
                        "pre_terminal_valve_angle_deg": math.degrees(
                            float(pre_terminal_evaluation["valve_angle"][env_id, 0].item())
                        ),
                        "pre_terminal_goal_error_normalized": goal_error_normalized,
                        "pre_terminal_goal_error_deg": goal_error_deg,
                    }
                )
                progress.update(1)
            counts[include] += 1
            episode_steps[done] = 0

            observation = policy_observation(observations)
            evaluation = evaluation_observation(observations, args_cli.num_envs)
            if not torch.isfinite(observation).all():
                raise RuntimeError(f"Observation contains NaN or Inf at step {total_steps}")
            history = torch.roll(history, shifts=-1, dims=1)
            history[:, -1] = observation
            if done.any():
                history[done] = observation[done].unsqueeze(1).repeat(
                    1, policy.n_obs_steps, 1
                )

        total_steps += 1
        if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
            break
        if args_cli.video and total_steps >= args_cli.video_length:
            break
        sleep_time = base_env.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    progress.close()
    env.close()

    episodes_path, summary_path = write_metrics(
        output_dir,
        records,
        {
            "policy_path": str(policy_path),
            "environment": TASK_ID,
            "task_direction": args_cli.task_direction,
            "success_threshold_deg": args_cli.success_threshold_deg,
            "requested_episodes": args_cli.num_episodes,
            "completed_episodes": len(records),
            "total_steps": total_steps,
            "num_envs": args_cli.num_envs,
            "observation_dim": policy.obs_dim,
            "action_dim": policy.action_dim,
            "control_frequency_hz": 1.0 / base_env.step_dt,
            "terminal_metrics_note": (
                "Valve metrics are sampled immediately before the terminating action "
                "because Isaac Lab resets completed environments before returning observations."
            ),
        },
    )
    successes = sum(bool(record["success"]) for record in records)
    print(
        f"[INFO] Evaluation complete: {successes}/{len(records)} successful "
        f"after {total_steps} steps."
    )
    print(f"[INFO] Episode metrics: {episodes_path}")
    print(f"[INFO] Summary: {summary_path}")
    print(f"[INFO] Successful rollouts: {output_dir / 'rollouts.hdf5'}")
    print(f"[INFO] Failed rollouts: {output_dir / 'rollouts_failed.hdf5'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
