#!/usr/bin/env python3
"""Precompute and cache reachable valve-first Spot reset states."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Precompute reachable Spot ball-valve reset states."
)
parser.add_argument(
    "--output_path",
    type=Path,
    required=True,
    help="Output .pt cache path.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Regenerate the cache even when the output path already exists.",
)
parser.add_argument(
    "--ik_batch_size",
    type=int,
    default=2048,
    help="Number of candidate valve states solved per IK batch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.ik_batch_size <= 0:
    parser.error("--ik_batch_size must be greater than zero")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab_hiveboard.tasks.spot.ball_valve.env import (
    SpotBallValveDeltaEnvCfg_PLAY,
)


TASK_ID = "Spot-Manipulation-Ball-Valve-Delta-Play"


def register_task() -> None:
    """Register the fixed-base Spot collection task."""
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


def main() -> None:
    """Construct the environment once to generate or load the reset cache."""
    register_task()
    env_cfg = SpotBallValveDeltaEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    cache_path = args_cli.output_path.expanduser().resolve()
    event_params = env_cfg.events.reset_robot_joints.params
    event_params["reset_state_cache_path"] = str(cache_path)
    event_params["reset_state_cache_force_regenerate"] = args_cli.force
    event_params["show_reset_state_progress"] = True
    event_params["ik_batch_size"] = args_cli.ik_batch_size

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        event = env.unwrapped.event_manager.get_term("reset_robot_joints")
        print(
            f"[INFO] Reset-state cache ready: {cache_path} "
            f"({event._num_valid_states} reachable states)."
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
