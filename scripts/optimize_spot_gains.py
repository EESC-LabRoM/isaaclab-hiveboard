#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate or optimize Spot arm/gripper PD gains on the robot-only tracking env.

The env replays the website joint clip in free space (no valve). Default
starting gains are the 4x hardware-style values on the bench Spot config.

Vectorized: ``--num-envs N`` scores N gain sets in one physics rollout.
``--optimize`` uses a (1+λ) log-space search with λ = num_envs.

Examples:

    uv run --python 3.12 python scripts/optimize_spot_gains.py \\
        --num-envs 16 physics=newton_mjwarp --visualizer none

    uv run --python 3.12 python scripts/optimize_spot_gains.py --optimize \\
        --joints all --num-envs 16 --max-evals 80 \\
        physics=newton_mjwarp --visualizer none
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab_hiveboard.assets.spot.bench import (
    ARM_DAMPING,
    ARM_JOINT_NAMES,
    ARM_STIFFNESS,
    apply_spot_arm_gains,
)
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


DEFAULT_TASK = "Isaac-HiveBoard-Spot-Gains-Play-v0"
JOINT_GROUPS = {
    "gripper": [6],
    "arm": list(range(6)),
    "all": list(range(7)),
}
# log10 bounds: gripper kp 1..316, kd 0.03..31; arm kp 10..1584, kd 0.3..100
_LOG_BOUNDS = {
    "gripper_kp": (0.0, 2.5),
    "gripper_kd": (-1.5, 1.5),
    "arm_kp": (1.0, 3.2),
    "arm_kd": (-0.5, 2.0),
}


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--optimize", action="store_true", help="Search kp/kd. Default: one eval.")
    parser.add_argument("--joints", choices=tuple(JOINT_GROUPS), default="gripper")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel environments (gain sets per rollout).")
    parser.add_argument("--max-evals", type=int, default=80, help="Total gain sets to score (optimize mode).")
    parser.add_argument("--sigma", type=float, default=0.18, help="Log10 perturbation std for the search.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kp",
        default=None,
        help="Comma-separated 7 stiffness values. Default: bench ARM_STIFFNESS.",
    )
    parser.add_argument(
        "--kd",
        default=None,
        help="Comma-separated 7 damping values. Default: bench ARM_DAMPING.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for JSON results. Default: logs/gain_opt/<timestamp>.",
    )
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    if args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        args.visualizer = ["none"]
    return args, hydra_args


def _parse_vector(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 7:
        raise SystemExit(f"Expected 7 comma-separated values, got {len(values)}: {text}")
    return np.asarray(values, dtype=np.float64)


def _as_numpy(value) -> np.ndarray:
    tensor = value.torch if hasattr(value, "torch") else value
    return tensor.detach().float().cpu().numpy()


def _clip_log(joint_id: int, log_kp: float, log_kd: float) -> tuple[float, float]:
    if joint_id == 6:
        lo_p, hi_p = _LOG_BOUNDS["gripper_kp"]
        lo_d, hi_d = _LOG_BOUNDS["gripper_kd"]
    else:
        lo_p, hi_p = _LOG_BOUNDS["arm_kp"]
        lo_d, hi_d = _LOG_BOUNDS["arm_kd"]
    return min(max(log_kp, lo_p), hi_p), min(max(log_kd, lo_d), hi_d)


def _pack_batch(base_kp, base_kd, joint_ids, x_log) -> tuple[np.ndarray, np.ndarray]:
    """x_log: (batch, 2 * len(joint_ids)) in log10 space."""
    batch = int(x_log.shape[0])
    kp = np.repeat(np.asarray(base_kp, dtype=np.float64)[None, :], batch, axis=0)
    kd = np.repeat(np.asarray(base_kd, dtype=np.float64)[None, :], batch, axis=0)
    for i, joint_id in enumerate(joint_ids):
        for env_i in range(batch):
            log_kp, log_kd = _clip_log(joint_id, float(x_log[env_i, 2 * i]), float(x_log[env_i, 2 * i + 1]))
            kp[env_i, joint_id] = 10.0 ** log_kp
            kd[env_i, joint_id] = 10.0 ** log_kd
    return kp, kd


def rollout_batch(env, stiffness: np.ndarray, damping: np.ndarray) -> list[dict]:
    """Score one gain set per environment. stiffness/damping: (num_envs, 7)."""
    base = env.unwrapped
    n_env = int(base.num_envs)
    stiffness = np.asarray(stiffness, dtype=np.float32)
    damping = np.asarray(damping, dtype=np.float32)
    if stiffness.ndim == 1:
        stiffness = np.repeat(stiffness[None, :], n_env, axis=0)
        damping = np.repeat(damping[None, :], n_env, axis=0)
    apply_spot_arm_gains(base.scene["robot"], stiffness, damping)
    obs, _ = env.reset()
    action_term = base.action_manager.get_term("arm_action")
    joint_ids = action_term._joint_ids
    robot = base.scene["robot"]
    steps = int(getattr(base, "max_episode_length", 0) or 0)
    if steps <= 0:
        step_dt = float(base.cfg.sim.dt) * float(base.cfg.decimation)
        steps = max(1, math.ceil(float(base.cfg.episode_length_s) / step_dt))

    sum_sq = np.zeros((n_env, 7), dtype=np.float64)
    max_abs = np.zeros((n_env, 7), dtype=np.float64)
    counts = np.zeros((n_env,), dtype=np.int32)
    alive = np.ones((n_env,), dtype=bool)

    for _ in range(steps):
        if isinstance(obs, dict) and isinstance(obs.get("policy"), dict) and "command" in obs["policy"]:
            action = obs["policy"]["command"]
        else:
            action = torch.zeros(env.action_space.shape, device=base.device)
        finite_action = torch.isfinite(action).all(dim=-1).cpu().numpy()
        with torch.no_grad():
            obs, _, terminated, truncated, _ = env.step(action)
        q_cmd = _as_numpy(action)
        q_act = _as_numpy(robot.data.joint_pos)[:, joint_ids]
        finite_state = np.isfinite(q_act).all(axis=-1)
        alive &= finite_action & finite_state
        err = q_act - q_cmd
        if alive.any():
            sum_sq[alive] += err[alive] ** 2
            max_abs[alive] = np.maximum(max_abs[alive], np.abs(err[alive]))
            counts[alive] += 1
        done = terminated if torch.is_tensor(terminated) else torch.as_tensor(terminated)
        trunc = truncated if torch.is_tensor(truncated) else torch.as_tensor(truncated)
        if bool((done | trunc).all()):
            break

    metrics = []
    for env_i in range(n_env):
        if counts[env_i] == 0:
            metrics.append(_failed_metrics(stiffness[env_i], damping[env_i], "non_finite"))
            continue
        rms = np.sqrt(sum_sq[env_i] / float(counts[env_i]))
        per_joint = {
            name: {
                "rms_rad": float(rms[i]),
                "rms_deg": float(np.rad2deg(rms[i])),
                "max_abs_rad": float(max_abs[env_i, i]),
                "max_abs_deg": float(np.rad2deg(max_abs[env_i, i])),
            }
            for i, name in enumerate(ARM_JOINT_NAMES)
        }
        metrics.append(
            {
                "ok": True,
                "n_steps": int(counts[env_i]),
                "stiffness": stiffness[env_i].astype(float).tolist(),
                "damping": damping[env_i].astype(float).tolist(),
                "rms_rad": rms.tolist(),
                "max_abs_rad": max_abs[env_i].tolist(),
                "mean_rms_rad": float(np.mean(rms)),
                "joints": per_joint,
            }
        )
    return metrics


def _failed_metrics(stiffness, damping, reason: str) -> dict:
    return {
        "ok": False,
        "reason": reason,
        "stiffness": np.asarray(stiffness, dtype=float).tolist(),
        "damping": np.asarray(damping, dtype=float).tolist(),
        "mean_rms_rad": 1.0e3,
        "joints": {},
    }


def _score(metrics: dict, joint_ids: list[int]) -> float:
    if not metrics.get("ok"):
        return 1.0e3
    rms = np.asarray(metrics["rms_rad"], dtype=np.float64)
    max_abs = np.asarray(metrics["max_abs_rad"], dtype=np.float64)
    penalty = float(np.sum(np.clip(max_abs[joint_ids] - 1.5, 0.0, None)))
    return float(np.mean(rms[joint_ids]) + 0.25 * penalty)


def _print_metrics(metrics: dict) -> None:
    print(f"[INFO] steps={metrics.get('n_steps')} mean_rms={metrics.get('mean_rms_rad'):.4f} rad")
    for name, stats in metrics.get("joints", {}).items():
        print(
            f"[INFO] {name}: rms={stats['rms_deg']:.2f} deg "
            f"max_abs={stats['max_abs_deg']:.2f} deg"
        )
    print(f"[INFO] kp={metrics['stiffness']}")
    print(f"[INFO] kd={metrics['damping']}")


def _to_log(kp, kd, joint_ids) -> np.ndarray:
    x = []
    for joint_id in joint_ids:
        x.extend(
            [
                math.log10(max(float(kp[joint_id]), 1e-3)),
                math.log10(max(float(kd[joint_id]), 1e-4)),
            ]
        )
    return np.asarray(x, dtype=np.float64)


def optimize(env, base_kp, base_kd, joint_ids, max_evals: int, sigma: float, seed: int) -> dict:
    """(1+λ) evolution in log10 gain space; λ = number of parallel envs."""
    rng = np.random.default_rng(seed)
    n_env = int(env.unwrapped.num_envs)
    dim = 2 * len(joint_ids)
    best_log = _to_log(base_kp, base_kd, joint_ids)
    history: list[dict] = []
    best: dict | None = None
    eval_count = 0
    generation = 0

    while eval_count < max_evals:
        generation += 1
        batch = min(n_env, max_evals - eval_count)
        x_log = np.repeat(best_log[None, :], batch, axis=0)
        if batch > 1:
            x_log[1:] = best_log[None, :] + sigma * rng.normal(size=(batch - 1, dim))
        kp, kd = _pack_batch(base_kp, base_kd, joint_ids, x_log)
        # Pad to num_envs if the last generation is smaller.
        if batch < n_env:
            kp_full = np.repeat(kp[:1], n_env, axis=0)
            kd_full = np.repeat(kd[:1], n_env, axis=0)
            kp_full[:batch] = kp
            kd_full[:batch] = kd
            metrics_all = rollout_batch(env, kp_full, kd_full)[:batch]
        else:
            metrics_all = rollout_batch(env, kp, kd)

        for metrics in metrics_all:
            eval_count += 1
            score = _score(metrics, joint_ids)
            metrics["score"] = score
            history.append(metrics)
            print(
                f"[EVAL {eval_count:03d}] gen={generation} score={score:.4f} "
                f"kp={[round(v, 3) for v in metrics['stiffness']]} "
                f"kd={[round(v, 3) for v in metrics['damping']]}",
                flush=True,
            )
            if best is None or score < best["score"]:
                best = metrics
                best_log = _to_log(metrics["stiffness"], metrics["damping"], joint_ids)
                print(f"[BEST] score={score:.4f}", flush=True)

    assert best is not None
    return {
        "method": "plus_lambda_log_es",
        "num_envs": n_env,
        "n_evals": len(history),
        "best": best,
        "history": history,
    }


def main() -> int:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = max(1, int(args.num_envs))
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    stiffness = _parse_vector(args.kp, ARM_STIFFNESS)
    damping = _parse_vector(args.kd, ARM_DAMPING)
    out_dir = args.out or os.path.join("logs", "gain_opt", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] num_envs={env_cfg.scene.num_envs}", flush=True)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            if args.optimize:
                payload = optimize(
                    env,
                    stiffness,
                    damping,
                    JOINT_GROUPS[args.joints],
                    args.max_evals,
                    args.sigma,
                    args.seed,
                )
                print("[INFO] Best:")
                _print_metrics(payload["best"])
            else:
                metrics = rollout_batch(env, stiffness, damping)[0]
                _print_metrics(metrics)
                payload = {"best": metrics, "n_evals": 1, "num_envs": env_cfg.scene.num_envs}
        finally:
            env.close()

    out_path = os.path.join(out_dir, "gains.json")
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[INFO] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
