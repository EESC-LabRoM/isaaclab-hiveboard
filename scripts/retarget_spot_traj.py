#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate the bench Spot clip from traj_edit beads using this robot's IK.

The vendored ``q`` was solved in MuJoCo against the website ``tcp`` site. This
script expands the Cartesian keys (same beads the markers use), tracks them
with damped-least-squares IK on the Isaac Lab Spot, and writes a new ``q``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import gymnasium as gym

import isaaclab_hiveboard  # noqa: F401
from isaaclab_hiveboard.assets.spot.bench import TRAJECTORY_JSON
from isaaclab_hiveboard.utils.spot_traj import IK_ITERS, load_payload, print_key_report, retarget, save_payload
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


DEFAULT_TASK = "Isaac-HiveBoard-Spot-Gains-Play-v0"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--traj", default=str(TRAJECTORY_JSON), help="Input trajectory JSON with keys + q.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON. Default: overwrite --traj (a .mujoco.json backup is kept).",
    )
    parser.add_argument("--ik-iters", type=int, default=IK_ITERS)
    parser.add_argument("--dry-run", action="store_true", help="Solve and print residuals; do not write.")
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    if args.visualizer is None and not getattr(args, "visualizer_explicit", False):
        args.visualizer = ["none"]
    return args, hydra_args


def main() -> int:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = 1
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders = None
    traj_path = Path(args.traj)
    payload = load_payload(traj_path)
    if "keys" not in payload or "q" not in payload:
        raise SystemExit(f"{traj_path} needs both 'keys' and 'q'.")

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            env.reset()
            q, key_report = retarget(env.unwrapped, payload, args.ik_iters)
        finally:
            env.close()

    print_key_report(key_report)
    if args.dry_run:
        return 0

    out_path = Path(args.out) if args.out else traj_path
    if out_path.resolve() == traj_path.resolve():
        backup = traj_path.with_suffix(".mujoco.json")
        if not backup.exists():
            shutil.copy2(traj_path, backup)
            print(f"[INFO] Backed up MuJoCo clip to {backup}")
    payload["q"] = q
    payload["source"] = f"retargeted from {traj_path.name} with Isaac Lab DLS IK"
    payload["keys_source"] = (
        "traj_edit beads + valve-arc finger/approach rotation; Isaac Lab pose IK on website tcp site"
    )
    save_payload(out_path, payload)
    print(f"[INFO] Wrote {len(q)} samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
