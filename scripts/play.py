# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified task player for HiveBoard manipulation environments (Spot, Franka)."""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play and debug HiveBoard manipulation tasks.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument(
    "--video_folder",
    type=str,
    default=os.path.join("videos", "play"),
    help="Directory for --video recordings (default: videos/play).",
)
parser.add_argument(
    "--video_name",
    type=str,
    default=None,
    help="Filename prefix for --video recordings (default: rl-video).",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help=(
        "Maximum recorded steps. Default: one episode, stopping on success or timeout so each clip is a single demo."
    ),
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=60,
    help="Playback and capture rate for --video (default: 60).",
)
parser.add_argument(
    "--video_width",
    type=int,
    default=2560,
    help="Recorded video width in pixels (default: 2560).",
)
parser.add_argument(
    "--video_height",
    type=int,
    default=1440,
    help="Recorded video height in pixels (default: 1440).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-HiveBoard-Spot-BallValve-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--orbit",
    dest="orbit",
    action="store_true",
    default=True,
    help="Orbit the viewport around the look-at point while playing (default: on).",
)
parser.add_argument(
    "--no-orbit",
    dest="orbit",
    action="store_false",
    help="Disable the camera orbit.",
)
parser.add_argument(
    "--orbit_deg",
    type=float,
    default=100.0,
    help="Yaw angle of the camera orbit in degrees.",
)
parser.add_argument(
    "--fast",
    action="store_true",
    default=False,
    help="Faster preview: 720p, 30 fps, DLSS, no GI/AO/reflections.",
)
parser.add_argument(
    "--pose-debug",
    action="store_true",
    default=False,
    help="Draw object/target/TCP frames and report tracking and physical alignment.",
)
parser.add_argument(
    "--pose-debug-interval",
    type=int,
    default=30,
    help="Print a compact pose report every N environment steps (default: 30).",
)
parser.add_argument(
    "--pose-debug-env",
    type=int,
    default=0,
    help="Environment index to inspect with --pose-debug (default: 0).",
)

parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="Stop after this many environment steps; useful for reproducible diagnostics.",
)
parser.add_argument(
    "--franka-position-only",
    action="store_true",
    default=False,
    help="Debug Franka translation by disabling the IK orientation objective.",
)
parser.add_argument(
    "--franka-breaker-pitch-deg",
    type=float,
    default=None,
    help=(
        "Debug a reachable Franka breaker pose by overriding target-frame pitch "
        "in the breaker source frame (configured default is -110)."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import torch
from tqdm import tqdm

import isaaclab_hiveboard  # noqa: F401


def main():
    """Main function."""
    if args_cli.task == "Isaac-HiveBoard-Spot-BallValve-v0" or args_cli.task == "Spot-Manipulation-Ball-Valve":
        from isaaclab_hiveboard.tasks.spot.ball_valve.env import SpotBallValveEnvCfg

        env_cfg = SpotBallValveEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Spot-BallValve-Play-v0":
        from isaaclab_hiveboard.tasks.spot.ball_valve.env import SpotBallValveEnvCfg_PLAY

        env_cfg = SpotBallValveEnvCfg_PLAY()
    elif args_cli.task == "Isaac-HiveBoard-Spot-BallValve-Play-Cameras-v0":
        from isaaclab_hiveboard.tasks.spot.ball_valve.env import SpotBallValveEnvCfg_PLAY_CAMERAS

        env_cfg = SpotBallValveEnvCfg_PLAY_CAMERAS()
    elif (
        args_cli.task == "Isaac-HiveBoard-Spot-HighTorqueValve-v0"
        or args_cli.task == "Spot-Manipulation-High-Torque-Valve"
    ):
        from isaaclab_hiveboard.tasks.spot.high_torque_valve.env import SpotHighTorqueValveEnvCfg

        env_cfg = SpotHighTorqueValveEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Spot-SmallValve-v0" or args_cli.task == "Spot-Manipulation-Small-Valve":
        from isaaclab_hiveboard.tasks.spot.small_valve.env import SpotSmallValveEnvCfg

        env_cfg = SpotSmallValveEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Spot-Button-v0" or args_cli.task == "Spot-Manipulation-Button":
        from isaaclab_hiveboard.tasks.spot.button.env import SpotButtonEnvCfg

        env_cfg = SpotButtonEnvCfg()
    elif (
        args_cli.task == "Isaac-HiveBoard-Spot-CircuitBreaker-v0"
        or args_cli.task == "Spot-Manipulation-Circuit-Breaker"
    ):
        from isaaclab_hiveboard.tasks.spot.circuit_breaker.env import SpotCircuitBreakerEnvCfg

        env_cfg = SpotCircuitBreakerEnvCfg()
    elif (
        args_cli.task == "Isaac-HiveBoard-Franka-CircuitBreaker-v0"
        or args_cli.task == "Franka-Manipulation-Circuit-Breaker"
    ):
        from isaaclab_hiveboard.tasks.franka.circuit_breaker.env import FrankaCircuitBreakerEnvCfg

        env_cfg = FrankaCircuitBreakerEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Franka-OnlyRobot-v0":
        from isaaclab_hiveboard.tasks.franka.only_robot.env import FrankaOnlyRobotEnvCfg

        env_cfg = FrankaOnlyRobotEnvCfg()
    elif args_cli.task in (
        "Isaac-HiveBoard-Franka-LeverValve-v0",
        "Franka-Manipulation-Lever-Valve",
        "Franka-Manipulation-Ball-Valve",
    ):
        from isaaclab_hiveboard.tasks.franka.lever_valve.env import FrankaLeverValveEnvCfg

        env_cfg = FrankaLeverValveEnvCfg()
    else:
        # Fallback to standard Gym registration if task is not in the explicit list
        print(f"[INFO] Using standard Gym registration for task: {args_cli.task}")
        env_cfg = None

    if env_cfg is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device

        if args_cli.pose_debug:
            if hasattr(env_cfg.scene, "target_frame"):
                env_cfg.scene.target_frame.debug_vis = True
            if hasattr(env_cfg.scene, "ee_frame"):
                env_cfg.scene.ee_frame.debug_vis = True
            # if hasattr(env_cfg.commands, "pose_command"):
            #     env_cfg.commands.pose_command.debug_vis = True

        if args_cli.franka_position_only:
            if not ("Franka" in args_cli.task):
                raise ValueError("--franka-position-only is only valid for Franka tasks")
            env_cfg.actions.arm_action.controller.command_type = "position"
            print("[INFO] Franka IK ablation: position-only control enabled.")

        if args_cli.franka_breaker_pitch_deg is not None:
            if not ("CircuitBreaker" in args_cli.task or "Circuit-Breaker" in args_cli.task):
                raise ValueError("--franka-breaker-pitch-deg is only valid for the Franka circuit-breaker task")
            pitch = math.radians(args_cli.franka_breaker_pitch_deg)
            quat_w = math.cos(pitch / 2.0)
            quat_y = math.sin(pitch / 2.0)
            env_cfg.commands.pose_command.target_frame_quat_in_source = (
                quat_w,
                0.0,
                quat_y,
                0.0,
            )
            print(
                "[INFO] Franka breaker IK ablation: source-frame target pitch "
                f"{args_cli.franka_breaker_pitch_deg:.1f} deg."
            )

        def _apply_fast_render() -> None:
            env_cfg.sim.render.antialiasing_mode = "DLSS"
            env_cfg.sim.render.dlss_mode = 0  # Performance
            env_cfg.sim.render.enable_reflections = False
            env_cfg.sim.render.enable_global_illumination = False
            env_cfg.sim.render.enable_ambient_occlusion = False
            env_cfg.sim.render.enable_dl_denoiser = False
            env_cfg.sim.render.enable_direct_lighting = True
            env_cfg.sim.render.enable_shadows = True
            env_cfg.sim.render.samples_per_pixel = 1

        if args_cli.fast:
            args_cli.video_width = 1280
            args_cli.video_height = 720
            args_cli.video_fps = 30
            _apply_fast_render()

        if args_cli.video:
            env_cfg.decimation = max(1, round((1.0 / env_cfg.sim.dt) / args_cli.video_fps))
            env_cfg.sim.render_interval = env_cfg.decimation
            env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
        else:
            # Default Isaac Lab render_interval is 1, so a 200 Hz sim draws the
            # viewport five times per env step (and vsync-locks to a crawl).
            env_cfg.sim.render_interval = env_cfg.decimation
            if not getattr(args_cli, "headless", False):
                env_cfg.viewer.resolution = (1280, 720)
                if not args_cli.fast:
                    _apply_fast_render()

        env = gym.make(
            id=args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
    else:
        env = gym.make(
            id=args_cli.task,
            render_mode="rgb_array" if args_cli.video else None,
        )

    base_env = env.unwrapped
    episode_steps = int(getattr(base_env, "max_episode_length", 0) or 0)
    if episode_steps <= 0:
        step_dt = float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
        episode_steps = max(1, math.ceil(float(base_env.cfg.episode_length_s) / step_dt))
    # One clip is one demo: stop on termination, cap at the episode horizon.
    record_steps = args_cli.video_length if args_cli.video_length is not None else episode_steps

    if args_cli.video:
        video_fps = 1.0 / (base_env.cfg.sim.dt * base_env.cfg.decimation)
        video_kwargs = {
            "video_folder": args_cli.video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": record_steps,
            "name_prefix": args_cli.video_name or "rl-video",
            "disable_logger": True,
            "fps": int(round(video_fps)),
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    count = 0
    obs, _ = env.reset()
    pbar = tqdm(total=record_steps) if args_cli.video else tqdm()

    diag_rows: list[dict] = []
    diag_interval = max(1, int(args_cli.pose_debug_interval))
    diag_names = None
    if "button" in base_env.scene.keys():
        button_asset = base_env.scene["button"]
        print("[play] button joint_names=", list(button_asset.joint_names))
        print("[play] button body_names=", list(button_asset.body_names))
        print(
            "[play] button root_pos=",
            button_asset.data.root_pos_w[0].detach().cpu().tolist(),
            "root_quat=",
            button_asset.data.root_quat_w[0].detach().cpu().tolist(),
        )
        for body in ("World", "lid_pivot", "button_pivot"):
            ids, names = button_asset.find_bodies(body)
            if ids:
                print(
                    f"[play] body {names[0]} pos=",
                    button_asset.data.body_pos_w[0, ids[0]].detach().cpu().tolist(),
                )
        frame = base_env.scene["target_frame"]
        diag_names = list(frame.data.target_frame_names)
        print("[play] target frames=", diag_names)
        for i, name in enumerate(diag_names):
            print(
                f"[play] frame {name} pos=",
                frame.data.target_pos_w[0, i].detach().cpu().tolist(),
            )
        ee = base_env.scene["ee_frame"]
        print(
            "[play] ee_tcp pos=",
            ee.data.target_pos_w[0, 0].detach().cpu().tolist(),
        )

    def _capture_diagnostics(step: int) -> None:
        diag = obs.get("diagnostics") if isinstance(obs, dict) else None
        if not isinstance(diag, dict):
            return
        row: dict = {"step": step}
        parts = [f"step={step}"]
        for key, value in diag.items():
            vals = value[0].detach().cpu().flatten().tolist()
            if len(vals) == 1:
                row[key] = vals[0]
                parts.append(f"{key}={vals[0]:.4f}")
            else:
                for i, item in enumerate(vals):
                    label = (
                        f"{key}_{diag_names[i]}"
                        if key == "ee_to_frames" and diag_names is not None and i < len(diag_names)
                        else f"{key}_{i}"
                    )
                    row[label] = item
                if key == "ee_to_frames" and diag_names is not None:
                    dist = " ".join(
                        f"{n}:{vals[i]:.3f}" for i, n in enumerate(diag_names) if i < len(vals)
                    )
                    parts.append(f"dist[{dist}]")
                elif key in ("lid_joint_pos", "button_joint_pos", "command_index"):
                    parts.append(f"{key}={vals}")
        diag_rows.append(row)
        if args_cli.pose_debug and (step % diag_interval == 0 or step <= 1):
            print("[play] " + " ".join(parts))

    _capture_diagnostics(0)

    action_terms = list(base_env.action_manager.active_terms) if hasattr(base_env, "action_manager") else []
    action_term_dims = list(base_env.action_manager.action_term_dim) if hasattr(base_env, "action_manager") else []

    relative_controller = None
    if set(action_terms) == {"gripper_action", "arm_action"} and 6 in action_term_dims:
        from isaaclab_hiveboard.mdp.relative_ee_pose_controller import RelativeEePoseController

        relative_controller = RelativeEePoseController(env)

    if action_terms == ["gripper_action", "arm_action"] and action_term_dims == [1, 7]:

        def _route_pose_command(command: torch.Tensor) -> torch.Tensor:
            return command
    elif action_terms == ["gripper_action", "arm_action"] and action_term_dims == [1, 3]:

        def _route_pose_command(command: torch.Tensor) -> torch.Tensor:
            return command[:, 0:4]
    elif action_terms == ["arm_action", "gripper_action"] and action_term_dims == [7, 1]:

        def _route_pose_command(command: torch.Tensor) -> torch.Tensor:
            return torch.cat((command[:, 1:8], command[:, 0:1]), dim=-1)
    elif action_terms == ["arm_action", "gripper_action"] and action_term_dims == [3, 1]:

        def _route_pose_command(command: torch.Tensor) -> torch.Tensor:
            return torch.cat((command[:, 1:4], command[:, 0:1]), dim=-1)
    else:

        def _route_pose_command(command: torch.Tensor) -> torch.Tensor:
            return command

    cam = getattr(base_env, "viewport_camera_controller", None)
    start_eye = tuple(base_env.cfg.viewer.eye) if hasattr(base_env.cfg, "viewer") else (2.0, 2.0, 1.0)
    lookat = tuple(base_env.cfg.viewer.lookat) if hasattr(base_env.cfg, "viewer") else (0.0, 0.0, 0.0)
    orbit_steps = record_steps if args_cli.video else episode_steps
    orbit_rad = math.radians(args_cli.orbit_deg)

    def _orbit_eye(step: int) -> tuple[float, float, float]:
        frac = min(step / max(orbit_steps - 1, 1), 1.0)
        yaw = orbit_rad * frac
        x, y, z = start_eye
        return (
            math.cos(yaw) * x - math.sin(yaw) * y,
            math.sin(yaw) * x + math.cos(yaw) * y,
            z,
        )

    def _apply_orbit(step: int) -> None:
        eye = _orbit_eye(step)
        if cam is not None:
            if getattr(base_env.cfg.viewer, "origin_type", None) == "asset_body":
                cam.update_view_to_asset_body(base_env.cfg.viewer.asset_name, base_env.cfg.viewer.body_name)
            cam.update_view_location(eye=eye, lookat=lookat)
            return
        origin = torch.zeros(3, device=base_env.device)
        if getattr(base_env.cfg.viewer, "origin_type", None) == "asset_body":
            asset = base_env.scene[base_env.cfg.viewer.asset_name]
            body_id = asset.find_bodies(base_env.cfg.viewer.body_name)[0][0]
            origin = asset.data.body_pos_w[base_env.cfg.viewer.env_index, body_id]
        eye_w = origin + origin.new_tensor(eye)
        target_w = origin + origin.new_tensor(lookat)
        base_env.sim.set_camera_view(
            eye=eye_w.detach().cpu().tolist(),
            target=target_w.detach().cpu().tolist(),
        )

    while simulation_app.is_running():
        with torch.inference_mode():
            if args_cli.orbit:
                _apply_orbit(count)

            if relative_controller is not None:
                action = relative_controller.compute()
            elif (
                isinstance(obs, dict)
                and isinstance(obs.get("policy"), dict)
                and "command" in obs["policy"]
            ):
                command = obs["policy"]["command"]
                action = _route_pose_command(command)
            else:
                action = torch.zeros(env.action_space.shape, device=base_env.device)

            obs, _, terminated, truncated, _ = env.step(action)
            count += 1
            pbar.update(1)
            _capture_diagnostics(count)

            term = terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)
            trunc = truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
            if term or trunc:
                print(f"[play] episode end at step {count} terminated={term} truncated={trunc}")
                term_mgr = getattr(base_env, "termination_manager", None)
                if term_mgr is not None:
                    for name in term_mgr.active_terms:
                        done = term_mgr.get_term(name)
                        flag = done.any().item() if torch.is_tensor(done) else bool(done)
                        print(f"[play]   termination.{name}={flag}")
                if "button" in base_env.scene.keys():
                    button = base_env.scene["button"]
                    print(
                        "[play]   button joint_pos=",
                        button.data.joint_pos[0].detach().cpu().tolist(),
                        "joint_names=",
                        list(button.joint_names),
                    )
                cmd = base_env.command_manager.get_term("pose_command")
                if hasattr(cmd, "_current_command_idx"):
                    print(
                        "[play]   command_index=",
                        int(cmd._current_command_idx[0].item()),
                        "/",
                        len(cmd.cfg.commands),
                    )
            if args_cli.video and (term or trunc or count >= record_steps):
                break
            if args_cli.max_steps is not None and count >= args_cli.max_steps:
                break

    if diag_rows:
        import csv

        diag_path = os.path.join("logs", "spot_button_diag.csv")
        os.makedirs("logs", exist_ok=True)
        with open(diag_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)
        print(f"[play] wrote {len(diag_rows)} diagnostic rows to {diag_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
