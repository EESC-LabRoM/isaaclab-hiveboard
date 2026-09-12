#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Probe sim-vs-URDF arm kinematics: nudge each arm joint, record flange motion.

Drives the robot-only Gains env headless, holds all joints except one nudged
arm joint at a time, and prints the world-frame flange (arm_link_wr1) pose
per probe. Compared offline against URDF FK, this identifies per-joint
axis/frame convention mismatches between the sim USD and the URDF.
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
NUDGE = 0.08
SETTLE_STEPS = 60
NUDGE_STEPS = 50
RESTORE_STEPS = 30


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
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

    def snapshot(robot, bidx):
        return {
            "flange_pos_w": robot.data.body_pos_w[0, bidx].detach().cpu().tolist(),
            "flange_quat_w": robot.data.body_quat_w[0, bidx].detach().cpu().tolist(),
            "root_pos_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
            "root_quat_w": robot.data.root_quat_w[0].detach().cpu().tolist(),
            "joint_pos": robot.data.joint_pos[0].detach().cpu().tolist(),
        }

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            base = env.unwrapped
            env.reset()
            robot = base.scene["robot"]
            joint_ids, joint_names = robot.find_joints(ARM_6, preserve_order=True)
            assert list(joint_names) == ARM_6, joint_names
            body_ids, _ = robot.find_bodies("arm_link_wr1")
            bidx = int(body_ids[0])
            print(json.dumps({"joint_names_all": list(robot.joint_names)}), flush=True)
            print(json.dumps({"arm_joint_ids": list(joint_ids)}), flush=True)

            home = robot.data.joint_pos[0, joint_ids].detach().clone()
            grip = torch.tensor([-1.5], device=base.device)
            resets = 0

            def step_hold(q6, n):
                nonlocal resets
                nonlocal home
                for _ in range(n):
                    action = torch.cat([q6, grip]).unsqueeze(0)
                    _, _, terminated, truncated, _ = env.step(action)
                    if bool(terminated.any()) or bool(truncated.any()):
                        resets += 1

            step_hold(home, SETTLE_STEPS)
            out = {"home": snapshot(robot, bidx), "home_q6": home.detach().cpu().tolist(), "probes": []}
            print(json.dumps({"home": out["home"]}), flush=True)

            for k, name in enumerate(ARM_6):
                nudged = home.clone()
                nudged[k] += NUDGE
                step_hold(nudged, NUDGE_STEPS)
                snap = snapshot(robot, bidx)
                snap.update({"joint": name, "delta": NUDGE})
                out["probes"].append(snap)
                print(json.dumps(snap), flush=True)
                step_hold(home, RESTORE_STEPS)
            print(json.dumps({"resets_during_probe": resets}), flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
