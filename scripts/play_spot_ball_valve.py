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
    if terms == ["arm_action", "gripper_action"] and dims == [6, 1]:
        if int(command.shape[-1]) == 7:
            return command
        return torch.cat((torch.zeros_like(command[:, 1:7]), command[:, 0:1]), dim=-1)
    raise RuntimeError(f"Unexpected action layout: terms={terms}, dimensions={dims}")


def _apply_collision_only(base) -> bool:
    """Hide visual geometry so Newton viewers show only collision shapes.

    Newton's viewer gates visual vs collision shapes independently
    (``show_visual``/``show_collision``). IsaacLab's ``NewtonVisualizerCfg``
    only exposes ``show_collision``, so the hide-visuals half has to be set
    on the live viewer object. No-op when no Newton viewer exists (e.g.
    ``--visualizer none``). Returns True if any viewer was switched.
    """
    switched = False
    for viz in base.sim.visualizers:
        viewer = getattr(viz, "_viewer", None)
        if viewer is None or not hasattr(viewer, "show_visual"):
            continue
        viewer.show_visual = False
        viewer.show_collision = True
        switched = True
    return switched


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument(
        "--collision-only",
        action="store_true",
        help="Newton visualizer shows only collision geometry (hides visual meshes). "
        "Config-driven, so it works headless with no viewer UI. No-op with --visualizer none.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Override episode length in sim-seconds (task default: 10 s = 200 steps). "
        "Longer values run one uninterrupted episode without an episode reset.",
    )
    parser.add_argument(
        "--valve-reset-angle",
        type=float,
        default=None,
        help="Fix the play-task valve reset angle [rad] (default: -0.35). "
        "The command samples the open/close direction from the reset state, "
        "so -1.5708 (fully open) yields a full-range close with 1.57 rad of "
        "travel instead of jamming into the upper stop after 0.35 rad.",
    )
    parser.add_argument(
        "--valve-friction",
        type=float,
        default=None,
        help="Override ball-valve joint friction [N·m, Newton absolute] "
        "(config default: 0.02). Use 0.0 for the zero-friction backdrivability test.",
    )
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
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = float(args.episode_length_s)
    if args.valve_reset_angle is not None:
        params = env_cfg.events.reset_valve_joint.params
        params["position_range"] = (float(args.valve_reset_angle), float(args.valve_reset_angle))
    if args.valve_friction is not None:
        env_cfg.scene.ball_valve.actuators["joint_actuator"].friction = float(args.valve_friction)
    if args.collision_only:
        try:
            from isaaclab_visualizers.newton import NewtonVisualizerCfg
        except ImportError as err:
            raise SystemExit(
                "--collision-only needs the Newton visualizer backend "
                "(pip install isaaclab_visualizers[newton])."
            ) from err
        env_cfg.sim.visualizer_cfgs = [NewtonVisualizerCfg(show_collision=True)]

    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    max_steps = args.max_steps or math.ceil(float(env_cfg.episode_length_s) / step_dt)
    success_tolerance = math.radians(15.0)

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        if args.collision_only and not _apply_collision_only(base):
            print(
                "[WARN] --collision-only: no Newton viewer active "
                "(use --visualizer newton, not none).",
                file=sys.stderr,
            )
        valve = base.scene["ball_valve"]
        valve_joint = valve.find_joints("RevoluteJoint")[0][0]
        obs, _ = env.reset(seed=args.seed)
        robot_root = base.scene["robot"].data.root_pos_w.torch[0].tolist()
        valve_root = valve.data.root_pos_w.torch[0].tolist()
        tcp = base.scene["ee_frame"].data.target_pos_w.torch[0, 0].tolist()
        command_term = base.command_manager.get_term("pose_command")
        tcp_b = command_term._get_ee_in_base_frame(slice(None))[0][0].tolist()
        # Success = reaching the sampled endpoint (open or close), so a reset
        # at the open stop (-pi/2) does not instantly "succeed" the old fixed
        # open-target check.
        valve_des = float(command_term.valve_joint_des[0].item())
        print(
            "RESET "
            f"robot_root={tuple(round(value, 4) for value in robot_root)} "
            f"valve_root={tuple(round(value, 4) for value in valve_root)} "
            f"tcp_sensor={tuple(round(value, 4) for value in tcp)} "
            f"tcp_b={tuple(round(value, 4) for value in tcp_b)} "
            f"valve_goal={tuple(round(value, 4) for value in (float(command_term.valve_task_goal[0].item()), valve_des))}"
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
                valve_qd = float(valve.data.joint_vel.torch[0, valve_joint].item())
                minimum_angle = min(minimum_angle, angle)
                if args.log_every > 0 and (step + 1) % args.log_every == 0:
                    command_term = base.command_manager.get_term("pose_command")
                    phase = int(command_term._current_command_idx[0].item())
                    tcp_b = command_term._get_ee_in_base_frame(slice(None))[0][0]
                    if phase < len(command_term._command_handlers):
                        handler = command_term._command_handlers[phase]
                        target_b = handler.get_target_in_base_frame(
                            torch.tensor([0], device=base.device)
                        )[0][0]
                        target_str = (
                            f"target_b={tuple(round(float(value), 4) for value in target_b)} "
                        )
                    else:
                        # Long episodes can run past the last waypoint; the
                        # command holds its final target, so there is nothing
                        # left to index.
                        target_str = "target_b=(sequence done) "
                    print(
                        f"STEP step={step + 1} phase={phase} "
                        f"tcp_b={tuple(round(float(value), 4) for value in tcp_b)} "
                        f"{target_str}"
                        f"valve_angle_rad={angle:.6f} "
                        f"valve_qd={valve_qd:.6f}"
                    )
                if abs(angle - valve_des) <= success_tolerance or bool(torch.as_tensor(terminated).any()):
                    print(f"SUCCESS step={step + 1} valve_angle_rad={angle:.6f}")
                    return 0
                if bool(torch.as_tensor(truncated).any()):
                    break
        finally:
            env.close()

    print(
        f"FAILURE valve did not reach {valve_des:.4f} rad +/- 15 deg; minimum_angle_rad={minimum_angle:.6f}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
