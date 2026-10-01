#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute a bank of cuRobo expert trajectories for the RL ball-valve task.

Runs the scripted cuRobo expert (``Isaac-HiveBoard-Anymal-BallValve-v0`` with
its saved command setup) in many parallel environments, with the RL task's
reset randomization (valve pose, initial valve angle, arm start), at the RL
task's 20 Hz control rate. cuRobo plans all environments entering a segment on
the same tick as one batch (``plan_batch_size``).

Environments run in synchronized waves: every episode lasts the same fixed
time and all reset together, so the planning batches stay large. A trajectory
is kept when the valve reaches open while the lever is held at the expert grasp
pose (the RL task's own success definition). The bank stores, per step, arm
joints and the expert's joint targets, gripper command and position, valve
angle, TCP pose and sequence phase, plus each episode's initial state::

    uv run python scripts/rl/build_expert_bank.py --num_envs 512 --num_trajectories 5000
"""

import argparse
import math
import os
import sys
import time

import gymnasium as gym
import isaaclab_hiveboard  # noqa: F401
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "imitation"))
from _common import _apply_command_setup  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

TASK = "Isaac-HiveBoard-Anymal-BallValve-v0"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--num_trajectories", type=int, default=5000, help="Successful trajectories to keep.")
parser.add_argument("--plan_batch_size", type=int, default=None, help="cuRobo batch (default: num_envs).")
parser.add_argument("--episode_length_s", type=float, default=13.0)
parser.add_argument("--max_waves", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output", default="logs/expert_bank/anymal_ball_valve_bank.pt")
parser.add_argument("--setup", default=None)
parser.add_argument("--no_setup", action="store_true")
parser.add_argument(
    "--video",
    default=None,
    help="Record environment 0 over all waves to this directory (Newton viewer). Use a few envs and waves.",
)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
if not any(t.startswith(("physics=", "presets=")) for t in hydra_args):
    hydra_args.append("physics=newton_mjwarp")
# Match the RL task's gripper ramp. A Hydra override survives the task parse;
# editing the action cfg in place before gym.make does not.
hydra_args.append("env.actions.gripper_action.close_speed=2.0")
args.visualizer = ["newton_gl"] if args.video else []
args.task = TASK
sys.argv = [sys.argv[0], *hydra_args]


def configure(env_cfg) -> dict:
    """Apply the RL task's randomization, rate and sizing to the scripted task."""
    from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_JOINT_NAMES
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import env as rl_env
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl.env import AnymalBallValveRLEnvCfg

    from isaaclab.managers import EventTermCfg, SceneEntityCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_tasks.core.cabinet import mdp as base_mdp

    rl = AnymalBallValveRLEnvCfg()
    # apply_setup rewrites the scene TCP frame with the setup's body offset,
    # which is rotated 90 deg from the RL task's TCP. The scripted command
    # plans with its own offset, so restoring the RL frame changes nothing the
    # expert does; it makes the recorded TCP poses and the hold check use the
    # RL task's convention.
    env_cfg.scene.ee_frame = rl.scene.ee_frame
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.decimation = rl.decimation
    env_cfg.episode_length_s = args.episode_length_s
    physics = env_cfg.sim.physics
    if hasattr(physics, "solver_cfg"):
        # nconmax is per world; the scripted task's 4000 only fits a few envs.
        physics.solver_cfg.nconmax = rl.sim.physics.newton_mjwarp.solver_cfg.nconmax
    # Same reset distribution as the RL task.
    ranges = {
        "valve_pose": dict(rl_env.VALVE_POSE_RANGE),
        "valve_angle": rl_env.VALVE_ANGLE_RANGE,
        "arm": rl_env.ARM_RANGE,
    }
    env_cfg.events.reset_valve_root.params["pose_range"] = dict(ranges["valve_pose"])
    env_cfg.events.reset_valve_joint.params["position_range"] = ranges["valve_angle"]
    env_cfg.events.reset_arm = EventTermCfg(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": ranges["arm"],
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True),
        },
    )
    # Fixed-length synchronized waves: no early termination, no HDF5 recorder.
    for name in ("success", "command_done"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)
    env_cfg.recorders = RecorderManagerBaseCfg()
    if args.video:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

        # Side view of the valve and arm, in env-0 coordinates.
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.55, 1.05, 1.05)
        env_cfg.viewer.lookat = (0.9, 0.0, 0.8)
        horizon = int(math.ceil(args.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)))
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=os.path.abspath(args.video), video_length=horizon * args.max_waves, video_interval=0
            )
        ]
    batch = args.plan_batch_size or args.num_envs
    for command in env_cfg.commands.pose_command.commands:
        if hasattr(command, "plan_batch_size"):
            command.plan_batch_size = batch
    return ranges


def main() -> None:
    from isaaclab_hiveboard.imitation.expert import ScriptedExpert
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import mdp as rl_mdp
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl.env import HOLD, SUCCESS_TOLERANCE_RAD
    from isaaclab_hiveboard.tasks.anymal.ball_valve_rl.expert_bank import GRIPPER_LINKAGE_JOINTS

    env_cfg, _ = resolve_task_config(TASK, "")
    _apply_command_setup(env_cfg, args)
    ranges = configure(env_cfg)

    with launch_simulation(env_cfg, args):
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        dev, n_envs = env.device, env.num_envs
        robot, valve = env.scene["robot"], env.scene["ball_valve"]
        arm_ids = robot.find_joints(env_cfg.actions.arm_action.joint_names, preserve_order=True)[0]
        grip_id = robot.find_joints(env_cfg.actions.gripper_action.joint_names)[0][0]
        # The whole Robotiq linkage, so episodes can start mid-grasp (reset_from_expert_bank).
        hand_ids, hand_names = robot.find_joints(GRIPPER_LINKAGE_JOINTS, preserve_order=True)
        command = env.command_manager.get_term("pose_command")
        horizon = int(env.max_episode_length)
        kept: dict[str, list[torch.Tensor]] = {}
        stats = {"waves": 0, "episodes": 0, "successes": 0, "fallback": 0, "kept": 0, "wall_s": 0.0}

        for wave in range(args.max_waves):
            t0 = time.perf_counter()
            env.reset()
            expert = ScriptedExpert(env)
            init = {
                "arm_q0": robot.data.joint_pos.torch[:, arm_ids].clone(),
                "valve_angle0": rl_mdp.valve_angle(env).clone(),
                "valve_pose_env": torch.cat(
                    (valve.data.root_pos_w.torch - env.scene.env_origins, valve.data.root_quat_w.torch), dim=-1
                ).clone(),
            }
            steps = {
                k: []
                for k in (
                    "arm_q",
                    "arm_target",
                    "gripper_cmd",
                    "gripper_q",
                    "valve_angle",
                    "tcp_pose",
                    "phase",
                    "pad_force",
                    "gripper_joint_pos",
                )
            }
            opened_held = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            ever_held = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            opened_any = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            best_progress = torch.zeros(n_envs, device=dev)
            min_grasp_err = torch.full((n_envs, 2), float("inf"), device=dev)
            t_open = torch.full((n_envs,), float("nan"), device=dev)
            fallback = torch.zeros(n_envs, dtype=torch.bool, device=dev)
            with torch.inference_mode():
                for t in range(horizon):
                    action = expert.compute()
                    steps["arm_q"].append(robot.data.joint_pos.torch[:, arm_ids].clone())
                    steps["arm_target"].append(action[:, :-1].clone())
                    steps["gripper_cmd"].append(action[:, -1].clone())
                    steps["gripper_q"].append(robot.data.joint_pos.torch[:, grip_id].clone())
                    steps["valve_angle"].append(rl_mdp.valve_angle(env).clone())
                    steps["tcp_pose"].append(rl_mdp.tcp_pose_b(env).clone())
                    steps["phase"].append(command._current_command_idx.clone())
                    # Valve-filtered contact force on each finger pad [N].
                    steps["pad_force"].append(rl_mdp.pad_valve_force(env).clone())
                    steps["gripper_joint_pos"].append(robot.data.joint_pos.torch[:, hand_ids].clone())
                    env.step(action)
                    held = rl_mdp.lever_held(env, HOLD["dist_threshold"], HOLD["ang_threshold"])
                    is_open = rl_mdp.valve_open_success(env, SUCCESS_TOLERANCE_RAD)
                    ever_held |= held
                    opened_any |= is_open
                    best_progress = torch.maximum(best_progress, rl_mdp.valve_progress(env))
                    dist, ang = rl_mdp.tcp_grasp_error(env)
                    min_grasp_err = torch.minimum(min_grasp_err, torch.stack((dist, ang), dim=-1))
                    now_open = is_open & held & ~opened_held
                    t_open[now_open] = (t + 1) * env.step_dt
                    opened_held |= now_open
                    fallback |= command.expert_fallback()
            wave_s = time.perf_counter() - t0
            # Keep only trajectories the expert ran as planned: a cuRobo
            # fallback (e.g. a chain that could not be planned as one motion)
            # is discarded even when it opened the valve.
            ok = opened_held & ~fallback
            stats["waves"] += 1
            stats["episodes"] += n_envs
            stats["successes"] += int(opened_held.sum())
            stats["fallback"] += int((opened_held & fallback).sum())
            stats["kept"] += int(ok.sum())
            stats["wall_s"] += wave_s
            for key, values in steps.items():
                kept.setdefault(key, []).append(torch.stack(values, dim=1)[ok].cpu())
            for key, value in init.items():
                kept.setdefault(key, []).append(value[ok].cpu())
            kept.setdefault("t_open", []).append(t_open[ok].cpu())
            kept.setdefault("expert_fallback", []).append(fallback[ok].cpu())
            total = stats["kept"]
            print(
                f"[BANK] wave {wave} diagnostics: ever held {int(ever_held.sum())}/{n_envs}, "
                f"opened by any means {int(opened_any.sum())}/{n_envs}, "
                f"mean best progress {best_progress.mean():.3f}, "
                f"median closest grasp error {min_grasp_err[:, 0].median() * 1000:.1f} mm / "
                f"{min_grasp_err[:, 1].median():.3f} rad",
                flush=True,
            )
            print(
                f"[BANK] wave {wave}: {int(opened_held.sum())}/{n_envs} opened while held, "
                f"{int((opened_held & fallback).sum())} discarded for a fallback, in {wave_s:.1f}s; kept {total}",
                flush=True,
            )
            if total >= args.num_trajectories:
                break

        bank = {key: torch.cat(values)[: args.num_trajectories] for key, values in kept.items()}
        bank["meta"] = {
            "task": TASK,
            "dt": env.step_dt,
            "horizon": horizon,
            "arm_joint_names": list(env_cfg.actions.arm_action.joint_names),
            "gripper_joint_names": list(hand_names),
            "ranges": ranges,
            "success_rate": stats["successes"] / max(stats["episodes"], 1),
            "stats": stats,
            "hold": dict(HOLD),
            "success_tolerance_rad": SUCCESS_TOLERANCE_RAD,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(bank, args.output)
        rate = stats["successes"] / max(stats["episodes"], 1)
        per_traj = stats["wall_s"] / max(stats["kept"], 1)
        print(
            f"[BANK] saved {len(bank['arm_q'])} trajectories to {args.output}\n"
            f"[BANK] expert success {stats['successes']}/{stats['episodes']} = {rate:.1%}, "
            f"{stats['fallback']} successes discarded for using a cuRobo fallback\n"
            f"[BANK] wall time {stats['wall_s']:.0f}s over {stats['waves']} waves = "
            f"{per_traj:.2f}s per kept trajectory",
            flush=True,
        )
        if len(bank["t_open"]):
            t = bank["t_open"]
            print(f"[BANK] time to open {t.mean():.2f} +/- {t.std():.2f}s", flush=True)
        env.close()


if __name__ == "__main__":
    main()
