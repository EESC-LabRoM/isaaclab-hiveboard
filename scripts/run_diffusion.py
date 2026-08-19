#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a standalone Diffusion Policy on the fixed-base Spot ball-valve task."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run a low-dimensional Diffusion Policy on the Spot ball valve."
)
parser.add_argument(
    "--policy_path",
    type=str,
    default=None,
    help="Standalone serialized Diffusion Policy checkpoint.",
)
parser.add_argument(
    "--diffusion_policy_path",
    type=str,
    default=None,
    help="Optional checkout to prepend to sys.path before loading the policy.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of parallel environments."
)
stopping_group = parser.add_mutually_exclusive_group()
stopping_group.add_argument(
    "--num_successes",
    type=int,
    default=None,
    help="Stop after this many successful episodes.",
)
stopping_group.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help=(
        "Stop after this many completed episodes and report the success rate. "
        "Episodes are distributed as evenly as possible across environments."
    ),
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Maximum number of environment steps before exiting.",
)
parser.add_argument(
    "--video", action="store_true", help="Record one video of policy execution."
)
parser.add_argument(
    "--video_length", type=int, default=600, help="Recorded video length in steps."
)
parser.add_argument(
    "--video_folder",
    type=str,
    default="videos/run_diffusion",
    help="Directory used for video output.",
)
parser.add_argument(
    "--metrics_dir",
    type=str,
    default="logs/diffusion_evaluations",
    help="Parent directory for per-episode CSV and aggregate JSON metrics.",
)
parser.add_argument(
    "--evaluation_name",
    type=str,
    default=None,
    help="Optional metrics run name; defaults to a timestamp.",
)
parser.add_argument(
    "--real-time", action="store_true", help="Pace execution at environment rate."
)
parser.add_argument(
    "--no_clip_actions",
    action="store_true",
    help="Do not clamp predicted normalized actions to [-1, 1].",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.policy_path is None:
    parser.error("--policy_path is required")
if args_cli.num_envs <= 0:
    parser.error("--num_envs must be greater than zero")
if args_cli.num_successes is not None and args_cli.num_successes <= 0:
    parser.error("--num_successes must be greater than zero")
if args_cli.num_episodes is not None and args_cli.num_episodes <= 0:
    parser.error("--num_episodes must be greater than zero")
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be greater than zero")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from isaaclab.utils.dict import print_dict
from tqdm import tqdm

from isaaclab_hiveboard.tasks.spot.ball_valve.env import (
    SpotBallValveDeltaEnvCfg_PLAY,
)
from isaaclab_hiveboard.utils.episode_metrics import (
    EpisodeMetricTracker,
    save_evaluation_metrics,
)


TASK_ID = "Spot-Manipulation-Ball-Valve-Diffusion-Play"
OBSERVATION_GROUP = "diffusion_policy"
OBSERVATION_DIM = 33
ACTION_DIM = 7
CONTROL_FREQUENCY_HZ = 20.0


def register_task() -> None:
    """Register the fixed-base diffusion evaluation task."""
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": (
                    "isaaclab_hiveboard.tasks.spot.ball_valve.env:"
                    "SpotBallValveDeltaEnvCfg_PLAY"
                )
            },
        )


def load_policy(policy_path: str, diffusion_policy_path: str | None, device: str):
    """Load a standalone Diffusion Policy checkpoint.

    Args:
        policy_path: Serialized policy path.
        diffusion_policy_path: Optional source checkout needed by serialized classes.
        device: Torch device used for checkpoint tensors.

    Returns:
        Loaded policy in evaluation mode.

    Raises:
        FileNotFoundError: If the policy or requested source checkout does not exist.
        RuntimeError: If the checkpoint cannot be deserialized.
    """
    checkpoint = Path(policy_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {checkpoint}")

    if diffusion_policy_path is not None:
        source_path = Path(diffusion_policy_path).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(
                f"Diffusion Policy checkout does not exist: {source_path}"
            )
        sys.path.insert(0, str(source_path))

    load_kwargs: dict[str, Any] = {
        "map_location": device,
        "weights_only": False,
    }

    import dill

    load_kwargs["pickle_module"] = dill

    try:
        policy = torch.load(checkpoint, **load_kwargs)
    except (ModuleNotFoundError, ImportError, AttributeError) as error:
        raise RuntimeError(
            "Could not import a class required by the policy checkpoint. Install "
            "the training environment or pass --diffusion_policy_path."
        ) from error
    policy.to(device)
    policy.eval()
    return policy


def patch_noise_scheduler(policy, device: str) -> None:
    """Apply compatibility defaults and keep scheduler tensors on the policy device.

    Args:
        policy: Loaded Diffusion Policy.
        device: Torch device used for inference.
    """
    if not hasattr(policy, "noise_scheduler"):
        return
    scheduler = policy.noise_scheduler
    if hasattr(scheduler, "_internal_dict"):
        config = dict(scheduler._internal_dict)
        defaults = {
            "timestep_spacing": "leading",
            "steps_offset": 0,
            "clip_sample_range": 1.0,
            "sample_max_value": 1.0,
            "thresholding": False,
            "dynamic_thresholding_ratio": 0.995,
            "rescale_betas_zero_snr": False,
        }
        if any(key not in config for key in defaults):
            try:
                from diffusers.configuration_utils import FrozenDict
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "The checkpoint needs scheduler compatibility patching, but "
                    "the 'diffusers' package is unavailable."
                ) from error
            for key, value in defaults.items():
                config.setdefault(key, value)
            scheduler._internal_dict = FrozenDict(config)
    for key, value in scheduler.__dict__.items():
        if isinstance(value, torch.Tensor):
            scheduler.__dict__[key] = value.to(device)
    original_set_timesteps = scheduler.set_timesteps
    scheduler.set_timesteps = lambda num_inference_steps, **kwargs: (
        original_set_timesteps(num_inference_steps, device=device, **kwargs)
    )


def validate_policy_contract(policy, observation: torch.Tensor) -> None:
    """Validate policy and live-environment tensor dimensions.

    Args:
        policy: Loaded Diffusion Policy.
        observation: Concatenated live observation, shape ``(num_envs, 33)``.

    Raises:
        RuntimeError: If the policy or environment violates the dataset contract.
    """
    if observation.ndim != 2 or observation.shape[-1] != OBSERVATION_DIM:
        raise RuntimeError(
            f"Live {OBSERVATION_GROUP!r} observation shape is "
            f"{tuple(observation.shape)}, expected (num_envs, {OBSERVATION_DIM})"
        )
    policy_observation_dim = getattr(policy, "obs_dim", None)
    if policy_observation_dim is not None and policy_observation_dim != OBSERVATION_DIM:
        raise RuntimeError(
            f"Policy obs_dim={policy_observation_dim}, expected {OBSERVATION_DIM}"
        )
    policy_action_dim = getattr(policy, "action_dim", None)
    if policy_action_dim is not None and policy_action_dim != ACTION_DIM:
        raise RuntimeError(
            f"Policy action_dim={policy_action_dim}, expected {ACTION_DIM}"
        )
    for attribute in ("n_obs_steps", "n_action_steps"):
        value = getattr(policy, attribute, None)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Policy has invalid {attribute}={value!r}")


def get_diffusion_observation(observations: dict[str, Any]) -> torch.Tensor:
    """Extract the concatenated diffusion observation from environment output.

    Args:
        observations: Observation dictionary returned by the environment.

    Returns:
        Concatenated observation tensor, shape ``(num_envs, 33)``.

    Raises:
        RuntimeError: If the group is absent or is not a tensor.
    """
    observation = observations.get(OBSERVATION_GROUP)
    if not isinstance(observation, torch.Tensor):
        raise RuntimeError(
            f"Environment did not return tensor observation group {OBSERVATION_GROUP!r}"
        )
    return observation


def get_evaluation_observation(
    observations: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract named physical signals used by the episode metric tracker."""
    evaluation = observations.get("evaluation")
    required = {
        "ee_pose_b": 7,
        "valve_angle": 1,
        "valve_velocity": 1,
        "object_root_pose_b": 7,
    }
    if not isinstance(evaluation, dict):
        raise RuntimeError("Environment did not return dictionary group 'evaluation'")
    for name, dimension in required.items():
        value = evaluation.get(name)
        if not isinstance(value, torch.Tensor) or value.shape != (
            args_cli.num_envs,
            dimension,
        ):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise RuntimeError(
                f"Evaluation observation {name!r} has shape {shape}, expected "
                f"({args_cli.num_envs}, {dimension})"
            )
    return evaluation


def main() -> None:
    """Run Diffusion Policy inference."""
    register_task()
    env_cfg = SpotBallValveDeltaEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # Evaluation should not silently create another training dataset.
    env_cfg.recorders = None

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
        print("[INFO] Recording policy video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    base_env = env.unwrapped
    control_frequency_hz = 1.0 / base_env.step_dt
    if abs(control_frequency_hz - CONTROL_FREQUENCY_HZ) > 1.0e-6:
        raise RuntimeError(
            f"Environment control frequency is {control_frequency_hz:.6g} Hz, "
            f"expected {CONTROL_FREQUENCY_HZ:.6g} Hz"
        )
    if list(base_env.action_manager.active_terms) != [
        "gripper_action",
        "arm_action",
    ] or list(base_env.action_manager.action_term_dim) != [1, 6]:
        raise RuntimeError(
            "Unexpected action-manager layout: "
            f"terms={list(base_env.action_manager.active_terms)}, "
            f"dims={list(base_env.action_manager.action_term_dim)}; expected "
            "gripper(1) followed by relative TCP delta(6)"
        )

    policy = load_policy(
        args_cli.policy_path, args_cli.diffusion_policy_path, base_env.device
    )
    patch_noise_scheduler(policy, base_env.device)

    observations, _ = env.reset()
    observation = get_diffusion_observation(observations)
    validate_policy_contract(policy, observation)
    if not torch.isfinite(observation).all():
        raise RuntimeError("Initial diffusion observation contains NaN or Inf")
    observation_history = observation.unsqueeze(1).repeat(1, policy.n_obs_steps, 1)
    evaluation_observation = get_evaluation_observation(observations)
    metric_tracker = (
        EpisodeMetricTracker(evaluation_observation, base_env.step_dt)
        if args_cli.num_episodes is not None
        else None
    )

    print(
        "[INFO] Diffusion contract: "
        f"control_frequency={control_frequency_hz:.1f} Hz, "
        f"observations={tuple(observation.shape)}, "
        f"history={tuple(observation_history.shape)}, "
        f"actions=({args_cli.num_envs}, {ACTION_DIM}), "
        f"n_action_steps={policy.n_action_steps}"
    )
    action_chunk = None
    chunk_step = policy.n_action_steps
    previous_done = torch.zeros(
        args_cli.num_envs, dtype=torch.bool, device=base_env.device
    )
    first_action = True
    step = 0
    successes = 0
    completed_episodes = 0
    episode_counts = torch.zeros(
        args_cli.num_envs, dtype=torch.long, device=base_env.device
    )
    episode_targets = None
    if args_cli.num_episodes is not None:
        episodes_per_env, remainder = divmod(args_cli.num_episodes, args_cli.num_envs)
        episode_targets = torch.full_like(episode_counts, episodes_per_env)
        episode_targets[:remainder] += 1
    progress = tqdm(desc="Running diffusion", unit="steps")

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            inference_ms = None
            if chunk_step >= policy.n_action_steps or previous_done.any():
                inference_start = time.perf_counter()
                result = policy.predict_action({"obs": observation_history})
                inference_ms = 1000.0 * (time.perf_counter() - inference_start)
                action_chunk = result.get("action")
                if not isinstance(action_chunk, torch.Tensor):
                    raise RuntimeError(
                        "policy.predict_action() returned no tensor 'action'"
                    )
                if action_chunk.ndim != 3 or action_chunk.shape[-1] != ACTION_DIM:
                    raise RuntimeError(
                        f"Policy action chunk shape {tuple(action_chunk.shape)} is invalid"
                    )
                chunk_step = 0

            chunk_boundary = chunk_step == 0
            raw_actions = action_chunk[:, chunk_step]
            chunk_step += 1
            if not torch.isfinite(raw_actions).all():
                raise RuntimeError(f"Policy produced NaN or Inf at step {step}")
            actions = (
                raw_actions
                if args_cli.no_clip_actions
                else raw_actions.clamp(-1.0, 1.0)
            )
            if first_action:
                print(
                    f"[INFO] First action chunk={tuple(action_chunk.shape)}, "
                    f"action[0]={actions[0].detach().cpu().tolist()}"
                )
                first_action = False

            if metric_tracker is not None:
                metric_tracker.record_step(
                    evaluation_observation,
                    actions,
                    raw_actions,
                    inference_ms,
                    chunk_boundary,
                )

            observations, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
            if episode_targets is None:
                successes += int(terminated.sum().item())
            else:
                count_episode = done & (episode_counts < episode_targets)
                successes += int((terminated & count_episode).sum().item())
                completed_episodes += int(count_episode.sum().item())
                episode_counts[count_episode] += 1
            observation = get_diffusion_observation(observations)
            next_evaluation_observation = get_evaluation_observation(observations)
            if metric_tracker is not None:
                metric_tracker.finish(done, terminated, count_episode)
                metric_tracker.reset(done, next_evaluation_observation)
            evaluation_observation = next_evaluation_observation
            if not torch.isfinite(observation).all():
                raise RuntimeError(
                    f"Diffusion observation contains NaN or Inf at step {step}"
                )
            observation_history = torch.roll(observation_history, shifts=-1, dims=1)
            observation_history[:, -1] = observation
            if done.any():
                observation_history[done] = (
                    observation[done].unsqueeze(1).repeat(1, policy.n_obs_steps, 1)
                )
            previous_done = done

        step += 1
        progress.update(1)
        if args_cli.num_episodes is None:
            progress.set_postfix(successes=successes)
        else:
            progress.set_postfix(
                completed=f"{completed_episodes}/{args_cli.num_episodes}",
                successes=successes,
            )
        if args_cli.num_successes is not None and successes >= args_cli.num_successes:
            break
        if (
            args_cli.num_episodes is not None
            and completed_episodes >= args_cli.num_episodes
        ):
            break
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            break
        if args_cli.video and step >= args_cli.video_length:
            break
        sleep_time = base_env.step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    progress.close()
    env.close()
    if args_cli.num_episodes is None:
        print(
            f"[INFO] Finished after {step} steps with {successes} successful episodes."
        )
    else:
        failures = completed_episodes - successes
        success_rate = (
            100.0 * successes / completed_episodes if completed_episodes > 0 else 0.0
        )
        summary = (
            "Evaluation complete"
            if completed_episodes >= args_cli.num_episodes
            else "Evaluation stopped early"
        )
        print(
            f"[INFO] {summary}: {successes}/{completed_episodes} successful "
            f"({success_rate:.1f}%), {failures} failed, after {step} steps."
        )
        evaluation_name = args_cli.evaluation_name or datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        output_dir = Path(args_cli.metrics_dir) / evaluation_name
        episodes_path, summary_path = save_evaluation_metrics(
            output_dir,
            metric_tracker.records,
            {
                "policy_path": str(Path(args_cli.policy_path).resolve()),
                "control_frequency_hz": control_frequency_hz,
                "num_envs": args_cli.num_envs,
                "requested_episodes": args_cli.num_episodes,
                "completed_episodes": completed_episodes,
                "steps": step,
                "n_obs_steps": policy.n_obs_steps,
                "n_action_steps": policy.n_action_steps,
                "observation_dim": OBSERVATION_DIM,
                "action_dim": ACTION_DIM,
                "terminal_state_note": (
                    "State-derived metrics use the last observation before the "
                    "terminating action because Isaac Lab resets completed "
                    "environments before returning observations."
                ),
            },
        )
        print(f"[INFO] Episode metrics: {episodes_path.resolve()}")
        print(f"[INFO] Metrics summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
