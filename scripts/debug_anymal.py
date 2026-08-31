#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless ANYmal valve play. Writes a tiny log for later inspection.

  logs/anymal_debug/summary.txt   one-screen setup + extrema
  logs/anymal_debug/traj.csv      step, cmd, tcp, err, valve

  uv run python scripts/debug_anymal.py --headless --device cuda:0 --max-steps 150
"""

from __future__ import annotations

import argparse
import csv
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compact ANYmal valve diagnostics.")
parser.add_argument("--task", type=str, default="Isaac-HiveBoard-Anymal-BallValve-v0")
parser.add_argument("--max-steps", type=int, default=150)
parser.add_argument("--log-every", type=int, default=15)
parser.add_argument("--out-dir", type=str, default="logs/anymal_debug")
parser.add_argument("--position-only", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
import isaaclab_hiveboard.tasks  # noqa: F401
from isaaclab_hiveboard.tasks.anymal.ball_valve.env import AnymalBallValveEnvCfg


def _axis(quat: torch.Tensor, vec: list[float]) -> list[float]:
    v = math_utils.quat_apply(quat.unsqueeze(0), quat.new_tensor([vec]))[0]
    return [round(float(v[i]), 3) for i in range(3)]


def main() -> None:
    env_cfg = AnymalBallValveEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    if args_cli.position_only:
        env_cfg.actions.arm_action.controller.command_type = "position"
        print("[INFO] position-only IK")
    env = gym.make(id=args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    valve = base.scene["ball_valve"]
    cmd_term = base.command_manager.get_term("pose_command")
    arm_ids, arm_names = robot.find_joints(list(base.cfg.actions.arm_action.joint_names))
    if not isinstance(arm_ids, torch.Tensor):
        arm_ids = torch.as_tensor(arm_ids, device=base.device, dtype=torch.long)
    vj, _ = valve.find_joints("RevoluteJoint")
    vj = vj[0] if vj else 0

    os.makedirs(args_cli.out_dir, exist_ok=True)
    setup = [
        f"fixed_base={robot.is_fixed_base}",
        f"n_joints={robot.num_joints} n_bodies={robot.num_bodies}",
        f"arm_joints={list(arm_names)}",
        f"ee_body={cmd_term._body_name} idx={cmd_term._body_idx}",
        f"root={ [round(float(x),3) for x in robot.data.root_pos_w[0]] }",
        f"valve={ [round(float(x),3) for x in valve.data.root_pos_w[0]] }",
        f"bodies={list(robot.data.body_names)}",
        f"all_joints={list(robot.data.joint_names)}",
    ]
    print("[SETUP]")
    for line in setup:
        print(" ", line)

    rows = []
    grasp_dumped = False
    obs, _ = env.reset()
    for step in range(args_cli.max_steps):
        with torch.inference_mode():
            command = obs["policy"]["command"]
            dim = int(env.action_space.shape[-1])
            action = command[:, :dim]
            valve_rad = float(valve.data.joint_pos[0, vj].item())
            env_ids = torch.tensor([0], device=base.device, dtype=torch.long)
            ee_p, ee_q = cmd_term._get_ee_in_base_frame(env_ids)
            cmd = cmd_term.command[0]
            cmd_p, cmd_q = cmd[1:4], cmd[4:8]
            idx = int(cmd_term._current_command_idx[0].item())
            pos_err = float(torch.linalg.vector_norm(ee_p[0] - cmd_p).item())
            ori_err = float(torch.rad2deg(math_utils.quat_error_magnitude(ee_q, cmd_q.unsqueeze(0))[0]).item())
            q = [round(float(x), 3) for x in robot.data.joint_pos[0, arm_ids].tolist()]
            obs, _, terminated, truncated, _ = env.step(action)
            row = {
                "step": step,
                "idx": idx,
                "grip": round(float(cmd[0].item()), 2),
                "pos_err": round(pos_err, 3),
                "ori_err": round(ori_err, 1),
                "ee": [round(float(x), 3) for x in ee_p[0].tolist()],
                "cmd": [round(float(x), 3) for x in cmd_p.tolist()],
                "tcpX": _axis(ee_q[0], [1, 0, 0]),
                "tcpZ": _axis(ee_q[0], [0, 0, 1]),
                "cmdX": _axis(cmd_q, [1, 0, 0]),
                "cmdZ": _axis(cmd_q, [0, 0, 1]),
                "valve": round(valve_rad, 3),
                "q": q,
            }
            rows.append(row)
            if step % args_cli.log_every == 0:
                print(
                    f"s={step:3d} i={idx} pos={pos_err:.3f} ori={ori_err:5.1f} "
                    f"ee={row['ee']} cmd={row['cmd']} "
                    f"tcpX={row['tcpX']} cmdX={row['cmdX']} v={valve_rad:+.2f}"
                )
            # Once seated on the lever, print palm / pad / valve heights so the
            # TCP offset can be nudged without guessing the 2F-140 approach tilt.
            if idx >= 1 and not grasp_dumped:
                grasp_dumped = True
                names = list(robot.data.body_names)

                def _body(name: str):
                    i = names.index(name)
                    p = [round(float(x), 4) for x in robot.data.body_pos_w[0, i].tolist()]
                    q = robot.data.body_quat_w[0, i]
                    return p, _axis(q, [1, 0, 0]), _axis(q, [0, 1, 0]), _axis(q, [0, 0, 1])

                palm_p, palm_x, palm_y, palm_z = _body("robotiq_base_link")
                lf_p, *_ = _body("left_inner_finger")
                rf_p, *_ = _body("right_inner_finger")
                vnames = list(valve.data.body_names)
                vi = vnames.index("alavanca_pivot")
                lever_p = [round(float(x), 4) for x in valve.data.body_pos_w[0, vi].tolist()]
                tcp_w_t, _ = cmd_term._get_ee_in_world_frame(env_ids)
                tcp_w = [round(float(x), 4) for x in tcp_w_t[0].tolist()]
                pad_mid = [(lf_p[i] + rf_p[i]) / 2 for i in range(3)]
                print("[GRASP]")
                print(f"  palm_w={palm_p}  +X={palm_x} +Y={palm_y} +Z={palm_z}")
                print(f"  left_inner_finger_w={lf_p}  right_inner_finger_w={rf_p}")
                print(f"  pad_mid_w={[round(x, 4) for x in pad_mid]}")
                print(f"  tcp_b={row['ee']} tcp_w={tcp_w} cmd_b={row['cmd']}")
                print(f"  alavanca_pivot_w={lever_p}")
                print(f"  dz_tcp-lever={tcp_w[2] - lever_p[2]:.4f}  dz_padmid-lever={pad_mid[2] - lever_p[2]:.4f}")
                print(f"  dz_palm-lever={palm_p[2] - lever_p[2]:.4f}")
            done = (terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)) or (
                truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
            )
            if done:
                print(f"[END] step={step}")
                break
    env.close()

    csv_path = os.path.join(args_cli.out_dir, "traj.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "idx", "grip", "pos_err", "ori_err", "ee", "cmd", "tcpX", "cmdX", "valve", "q"])
        for r in rows:
            writer.writerow(
                [r["step"], r["idx"], r["grip"], r["pos_err"], r["ori_err"], r["ee"], r["cmd"], r["tcpX"], r["cmdX"], r["valve"], r["q"]]
            )

    pos = [r["pos_err"] for r in rows]
    ori = [r["ori_err"] for r in rows]
    summary = os.path.join(args_cli.out_dir, "summary.txt")
    lines = setup + [
        f"steps={len(rows)} last_idx={rows[-1]['idx']}",
        f"pos_err min/max={min(pos):.3f}/{max(pos):.3f} last={pos[-1]:.3f}",
        f"ori_err min/max={min(ori):.1f}/{max(ori):.1f} last={ori[-1]:.1f}",
        f"first ee={rows[0]['ee']} tcpX={rows[0]['tcpX']} tcpZ={rows[0]['tcpZ']}",
        f"first cmd={rows[0]['cmd']} cmdX={rows[0]['cmdX']} cmdZ={rows[0]['cmdZ']}",
        f"last  ee={rows[-1]['ee']} tcpX={rows[-1]['tcpX']}",
        f"last  cmd={rows[-1]['cmd']} cmdX={rows[-1]['cmdX']}",
        f"valve 0→{rows[-1]['valve']}",
        f"q0={rows[0]['q']}",
        f"qN={rows[-1]['q']}",
    ]
    Path_write = "\n".join(lines) + "\n"
    open(summary, "w").write(Path_write)
    print("[SUMMARY]")
    print(Path_write)


if __name__ == "__main__":
    main()
