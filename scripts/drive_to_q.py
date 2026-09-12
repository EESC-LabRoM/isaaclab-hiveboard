#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Drive the sim arm to offline-verified cuRobo plan-end joints, measure EE.

Reads /tmp/opencode/end_q.json (plan endpoint proven exact offline) and holds
it with absolute joint targets. If the EE lands on the bead target, sim-side
execution is vindicated and the sim-run plans were bad. If not, execution or
far-from-home kinematics diverge.
"""

from __future__ import annotations

import argparse
import json
import sys

import gymnasium as gym
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

DEFAULT_TASK = "Isaac-HiveBoard-Spot-Gains-Play-v0"
ARM_6 = list(ARM_JOINT_NAMES[:-1])


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--qjson", default="/tmp/opencode/end_q.json")
    parser.add_argument("--hold-steps", type=int, default=150)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    if args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        args.visualizer = []
        args.visualizer_explicit = True
    return args, hydra_args


def main() -> int:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders = None
    end_q = torch.tensor(json.load(open(args.qjson))["end_q"], dtype=torch.float32)
    assert end_q.numel() == 6, end_q.shape

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            base = env.unwrapped
            env.reset()
            robot = base.scene["robot"]
            joint_ids, _ = robot.find_joints(ARM_6, preserve_order=True)
            body_ids, _ = robot.find_bodies("arm_link_wr1")
            bidx = int(body_ids[0])
            grip = torch.tensor([-1.5], device=base.device)

            home = robot.data.joint_pos[0, joint_ids].detach().clone()
            for _ in range(60):
                env.step(torch.cat([home, grip]).unsqueeze(0))
            for _ in range(args.hold_steps):
                env.step(torch.cat([end_q.to(base.device), grip]).unsqueeze(0))

            print(
                json.dumps({
                    "commanded_q6": end_q.tolist(),
                    "actual_q6": robot.data.joint_pos[0, joint_ids].detach().cpu().tolist(),
                    "flange_pos_w": robot.data.body_pos_w[0, bidx].detach().cpu().tolist(),
                    "flange_quat_w": robot.data.body_quat_w[0, bidx].detach().cpu().tolist(),
                    "root_pos_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
                    "root_quat_w": robot.data.root_quat_w[0].detach().cpu().tolist(),
                }),
                flush=True,
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
