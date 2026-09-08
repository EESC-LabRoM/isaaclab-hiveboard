#!/usr/bin/env python3
"""Run and validate the kitless Isaac Lab 3 Spot ball-valve demo."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping

import gymnasium as gym
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


DEFAULT_TASK = "Isaac-HiveBoard-Spot-BallValve-Play-v0"


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
    if terms == ["arm_action", "gripper_action"] and dims == [7, 1]:
        return torch.cat((command[:, 1:8], command[:, 0:1]), dim=-1)
    raise RuntimeError(f"Unexpected action layout: terms={terms}, dimensions={dims}")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
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
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device

    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    max_steps = args.max_steps or math.ceil(float(env_cfg.episode_length_s) / step_dt)
    success_limit = -math.pi / 2 + math.radians(15.0)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        valve = base.scene["ball_valve"]
        valve_joint = valve.find_joints("RevoluteJoint")[0][0]
        obs, _ = env.reset(seed=args.seed)
        robot_root = base.scene["robot"].data.root_pos_w.torch[0].tolist()
        valve_root = valve.data.root_pos_w.torch[0].tolist()
        tcp = base.scene["ee_frame"].data.target_pos_w.torch[0, 0].tolist()
        command_term = base.command_manager.get_term("pose_command")
        tcp_b = command_term._get_ee_in_base_frame(slice(None))[0][0].tolist()
        print(
            "RESET "
            f"robot_root={tuple(round(value, 4) for value in robot_root)} "
            f"valve_root={tuple(round(value, 4) for value in valve_root)} "
            f"tcp_sensor={tuple(round(value, 4) for value in tcp)} "
            f"tcp_b={tuple(round(value, 4) for value in tcp_b)}"
        )
        minimum_angle = float("inf")

        try:
            for step in range(max_steps):
                if not _all_finite(obs):
                    raise FloatingPointError(f"Non-finite observation at step {step}")
                command = obs["policy"]["command"]
                action = _route_command(base, command)
                if not _all_finite(action):
                    raise FloatingPointError(f"Non-finite action at step {step}")
                obs, _, terminated, truncated, _ = env.step(action)
                angle = float(valve.data.joint_pos.torch[0, valve_joint].item())
                minimum_angle = min(minimum_angle, angle)
                if args.log_every > 0 and (step + 1) % args.log_every == 0:
                    command_term = base.command_manager.get_term("pose_command")
                    phase = int(command_term._current_command_idx[0].item())
                    tcp_b = command_term._get_ee_in_base_frame(slice(None))[0][0]
                    handler = command_term._command_handlers[phase]
                    target_b = handler.get_target_in_base_frame(
                        torch.tensor([0], device=base.device)
                    )[0][0]
                    print(
                        f"STEP step={step + 1} phase={phase} "
                        f"tcp_b={tuple(round(float(value), 4) for value in tcp_b)} "
                        f"target_b={tuple(round(float(value), 4) for value in target_b)} "
                        f"valve_angle_rad={angle:.6f}"
                    )
                if angle <= success_limit or bool(torch.as_tensor(terminated).any()):
                    print(f"SUCCESS step={step + 1} valve_angle_rad={angle:.6f}")
                    return 0
                if bool(torch.as_tensor(truncated).any()):
                    break
        finally:
            env.close()

    print(
        f"FAILURE valve did not reach -90 deg +/- 15 deg; minimum_angle_rad={minimum_angle:.6f}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
