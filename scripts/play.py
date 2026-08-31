# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unified task player for HiveBoard manipulation environments (Spot, Franka, ANYmal)."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

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
    "--ee-debug",
    action="store_true",
    default=False,
    help="Print gripper-body vs offset-TCP axes once after reset (tune ANYMAL_EE.tcp_offset).",
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
    "--lamp-distance-plot",
    type=str,
    default=None,
    help=(
        "Write a PNG and CSV trace of TCP distance from the lamp screw frame; "
        "intended for the Spot lamp task."
    ),
)
parser.add_argument(
    "--contact-debug",
    action="store_true",
    default=False,
    help="Draw red markers at gripper/lamp contact points; markers are included in --video output.",
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

import isaaclab.utils.math as math_utils
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
    elif (
        args_cli.task == "Isaac-HiveBoard-Spot-Lamp-v0"
        or args_cli.task == "Spot-Manipulation-Lamp"
    ):
        from isaaclab_hiveboard.tasks.spot.lamp.env import SpotLampEnvCfg

        env_cfg = SpotLampEnvCfg()
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
    elif args_cli.task == "Isaac-HiveBoard-Anymal-OnlyRobot-v0":
        from isaaclab_hiveboard.tasks.anymal.only_robot.env import AnymalOnlyRobotEnvCfg

        env_cfg = AnymalOnlyRobotEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Anymal-OnlyGripper-v0":
        from isaaclab_hiveboard.tasks.anymal.only_gripper.env import AnymalOnlyGripperEnvCfg

        env_cfg = AnymalOnlyGripperEnvCfg()
    elif args_cli.task in (
        "Isaac-HiveBoard-Anymal-BallValve-v0",
        "Anymal-Manipulation-Ball-Valve",
    ):
        from isaaclab_hiveboard.tasks.anymal.ball_valve.env import AnymalBallValveEnvCfg

        env_cfg = AnymalBallValveEnvCfg()
    elif args_cli.task in (
        "Isaac-HiveBoard-Franka-LeverValve-v0",
        "Franka-Manipulation-Lever-Valve",
        "Franka-Manipulation-Ball-Valve",
    ):
        from isaaclab_hiveboard.tasks.franka.lever_valve.env import FrankaLeverValveEnvCfg

        env_cfg = FrankaLeverValveEnvCfg()
    elif args_cli.task == "Isaac-HiveBoard-Franka-Lamp-v0":
        from isaaclab_hiveboard.tasks.franka.lamp.env import FrankaLampEnvCfg

        env_cfg = FrankaLampEnvCfg()
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

        if args_cli.contact_debug:
            contact_sensor_names = ("finger_contact", "jaw_contact")
            missing = [name for name in contact_sensor_names if not hasattr(env_cfg.scene, name)]
            if missing:
                raise ValueError("--contact-debug is only supported by tasks with gripper contact sensors")
            for name in contact_sensor_names:
                sensor_cfg = getattr(env_cfg.scene, name)
                sensor_cfg.track_contact_points = True
                sensor_cfg.max_contact_data_count_per_prim = 16

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

        if args_cli.fast:
            args_cli.video_width = 1280
            args_cli.video_height = 720
            args_cli.video_fps = 30
            env_cfg.sim.render.antialiasing_mode = "DLSS"
            env_cfg.sim.render.dlss_mode = 0  # Performance
            env_cfg.sim.render.enable_reflections = False
            env_cfg.sim.render.enable_global_illumination = False
            env_cfg.sim.render.enable_ambient_occlusion = False
            env_cfg.sim.render.enable_dl_denoiser = False
            env_cfg.sim.render.enable_direct_lighting = True
            env_cfg.sim.render.enable_shadows = True
            env_cfg.sim.render.samples_per_pixel = 1

        if args_cli.video:
            env_cfg.decimation = max(1, round((1.0 / env_cfg.sim.dt) / args_cli.video_fps))
            env_cfg.sim.render_interval = env_cfg.decimation
            env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)

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
    if args_cli.ee_debug or args_cli.pose_debug or "OnlyRobot" in args_cli.task or "OnlyGripper" in args_cli.task:
        from isaaclab_hiveboard.assets.end_effector import (
            ANYMAL_EE,
            FRANKA_EE,
            SPOT_EE,
            print_ee_offset_report,
        )

        if "OnlyGripper" in args_cli.task:
            from isaaclab_hiveboard.tasks.anymal.only_gripper.env import ROBOTIQ_DEBUG_EE

            print_ee_offset_report(base_env, ROBOTIQ_DEBUG_EE)
        elif "Anymal" in args_cli.task:
            print_ee_offset_report(base_env, ANYMAL_EE)
        elif "Franka" in args_cli.task:
            print_ee_offset_report(base_env, FRANKA_EE)
        elif "Spot" in args_cli.task:
            print_ee_offset_report(base_env, SPOT_EE)
    pbar = tqdm(total=record_steps) if args_cli.video else tqdm()

    contact_visualizer = None
    contact_sensors = ()
    if args_cli.contact_debug:
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        contact_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/LampContactPoints",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=0.012,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1.0, 0.0, 0.0),
                            emissive_color=(1.0, 0.0, 0.0),
                        ),
                    ),
                    "no_contact": sim_utils.SphereCfg(
                        radius=0.012,
                        visible=False,
                    ),
                },
            )
        )
        contact_sensors = tuple(
            base_env.scene[name] for name in ("finger_contact", "jaw_contact")
        )

    def _update_contact_visualizer() -> None:
        if contact_visualizer is None:
            return
        positions = []
        marker_indices = []
        debug_env = min(args_cli.pose_debug_env, base_env.num_envs - 1)
        for sensor in contact_sensors:
            contact_pos_w = sensor.data.contact_pos_w[debug_env].reshape(-1, 3)
            force_matrix_w = sensor.data.force_matrix_w[debug_env].reshape(-1, 3)
            in_contact = torch.linalg.vector_norm(force_matrix_w, dim=-1) > 0.1
            valid = in_contact & torch.isfinite(contact_pos_w).all(dim=-1)
            positions.append(torch.nan_to_num(contact_pos_w))
            marker_indices.append(torch.where(valid, 0, 1))
        contact_visualizer.visualize(
            torch.cat(positions, dim=0),
            marker_indices=torch.cat(marker_indices, dim=0),
        )

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

    lamp_distance_rows = []
    lamp_trace = None
    if args_cli.lamp_distance_plot is not None:
        if "lamp" not in base_env.scene.keys() or "target_frame" not in base_env.scene.keys():
            raise ValueError("--lamp-distance-plot is only supported by the lamp task")
        command_term = base_env.command_manager.get_term("pose_command")
        target_frame = base_env.scene["target_frame"]
        screw_frame_idx = target_frame.data.target_frame_names.index("screw_frame")
        lamp = base_env.scene["lamp"]
        robot = base_env.scene["robot"]
        revolute_idx = lamp.find_joints("RevoluteJoint")[0][0]
        prismatic_idx = lamp.find_joints("PrismaticJoint")[0][0]
        lamp_body_idx = lamp.find_bodies("lamp_pivot")[0][0]
        gripper_joint_ids = robot.find_joints("arm_f1x")[0]
        gripper_joint_idx = gripper_joint_ids[0] if gripper_joint_ids else None
        lamp_trace = (
            command_term,
            target_frame,
            screw_frame_idx,
            lamp,
            robot,
            revolute_idx,
            prismatic_idx,
            lamp_body_idx,
            gripper_joint_idx,
        )

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
            _update_contact_visualizer()

            if args_cli.orbit:
                _apply_orbit(count)

            if relative_controller is not None:
                action = relative_controller.compute()
            elif args_cli.task == "Isaac-HiveBoard-Anymal-OnlyGripper-v0":
                from isaaclab_hiveboard.assets import ROBOTIQ_CLOSE_Q, ROBOTIQ_JOINT_GEAR, robotiq_joint_targets

                # 4 s open→close→open. Closed is 0.7 rad. Inner fingers track -q.
                period_s = 4.0
                t = count * float(base_env.cfg.sim.dt) * float(base_env.cfg.decimation)
                finger_cmd = 0.5 * ROBOTIQ_CLOSE_Q * (1.0 - math.cos(2.0 * math.pi * t / period_s))
                targets = robotiq_joint_targets(finger_cmd)
                action = torch.zeros(env.action_space.shape, device=base_env.device)
                for i, name in enumerate(ROBOTIQ_JOINT_GEAR):
                    action[:, i] = targets[name]
                if count % 30 == 0:
                    robot = base_env.scene["robot"]
                    bits = [f"cmd={finger_cmd:.3f}"]
                    for name, value in zip(robot.data.joint_names, robot.data.joint_pos[0], strict=True):
                        bits.append(f"{name}={float(value):+.3f}")
                    print("[gripper] " + "  ".join(bits))
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

            if lamp_trace is not None:
                (
                    trace_command,
                    target_frame,
                    screw_frame_idx,
                    lamp,
                    robot,
                    revolute_idx,
                    prismatic_idx,
                    lamp_body_idx,
                    gripper_joint_idx,
                ) = lamp_trace
                trace_env = min(args_cli.pose_debug_env, base_env.num_envs - 1)
                env_ids = torch.tensor([trace_env], device=base_env.device)
                ee_pos_w, ee_quat_w = trace_command._get_ee_in_world_frame(env_ids)
                lamp_pos_w = target_frame.data.target_pos_w[
                    env_ids, screw_frame_idx
                ]
                lamp_quat_w = target_frame.data.target_quat_w[
                    env_ids, screw_frame_idx
                ]
                ee_pos_l, _ = math_utils.subtract_frame_transforms(
                    lamp_pos_w, lamp_quat_w, ee_pos_w, ee_quat_w
                )
                rel = ee_pos_l[0]
                radial = torch.linalg.vector_norm(rel[1:])
                total = torch.linalg.vector_norm(rel)
                command_idx = int(trace_command._current_command_idx[trace_env].item())
                handler_name = (
                    type(trace_command._command_handlers[command_idx]).__name__
                    if command_idx < len(trace_command._command_handlers)
                    else "done"
                )
                lamp_lin_vel = float(
                    torch.linalg.vector_norm(
                        lamp.data.body_lin_vel_w[trace_env, lamp_body_idx]
                    ).item()
                )
                lamp_ang_vel = float(
                    torch.linalg.vector_norm(
                        lamp.data.body_ang_vel_w[trace_env, lamp_body_idx]
                    ).item()
                )
                finger_n = float("nan")
                jaw_n = float("nan")
                for name, key in (("finger_contact", "finger_n"), ("jaw_contact", "jaw_n")):
                    if name not in base_env.scene.keys():
                        continue
                    force = torch.linalg.vector_norm(
                        base_env.scene[name].data.force_matrix_w[trace_env], dim=-1
                    ).max()
                    if key == "finger_n":
                        finger_n = float(force.item())
                    else:
                        jaw_n = float(force.item())
                gripper_rad = float("nan")
                if gripper_joint_idx is not None:
                    gripper_rad = float(
                        robot.data.joint_pos[trace_env, gripper_joint_idx].item()
                    )
                lamp_distance_rows.append(
                    {
                        "step": count,
                        "time_s": count * float(base_env.step_dt),
                        "command_idx": command_idx,
                        "command_type": handler_name,
                        "axial_mm": float(rel[0].item() * 1000.0),
                        "radial_mm": float(radial.item() * 1000.0),
                        "distance_mm": float(total.item() * 1000.0),
                        "lamp_angle_rad": float(
                            lamp.data.joint_pos[trace_env, revolute_idx].item()
                        ),
                        "lamp_insertion_mm": float(
                            lamp.data.joint_pos[trace_env, prismatic_idx].item()
                            * 1000.0
                        ),
                        "lamp_revolute_vel": float(
                            lamp.data.joint_vel[trace_env, revolute_idx].item()
                        ),
                        "lamp_prismatic_vel": float(
                            lamp.data.joint_vel[trace_env, prismatic_idx].item()
                        ),
                        "lamp_lin_vel": lamp_lin_vel,
                        "lamp_ang_vel": lamp_ang_vel,
                        "gripper_rad": gripper_rad,
                        "finger_contact_N": finger_n,
                        "jaw_contact_N": jaw_n,
                        "finite": int(
                            math.isfinite(lamp_lin_vel)
                            and math.isfinite(float(total.item()))
                        ),
                    }
                )

            if args_cli.pose_debug and count % args_cli.pose_debug_interval == 0:
                if contact_sensors:
                    contact_force_text = {
                        name: {
                            "net_N": float(
                                torch.linalg.vector_norm(sensor.data.net_forces_w[0], dim=-1)
                                .max()
                                .item()
                            ),
                            "lamp_N": float(
                                torch.linalg.vector_norm(sensor.data.force_matrix_w[0], dim=-1)
                                .max()
                                .item()
                            ),
                        }
                        for name, sensor in zip(
                            ("finger", "jaw"), contact_sensors
                        )
                    }
                    print(f"[CONTACT] step={count} {contact_force_text}", flush=True)
                command_term = base_env.command_manager.get_term("pose_command")
                command_idx = getattr(command_term, "_current_command_idx", None)
                debug_env = min(args_cli.pose_debug_env, base_env.num_envs - 1)
                tracked_name = getattr(base_env.cfg.viewer, "asset_name", None)
                tracked = base_env.scene[tracked_name] if tracked_name in base_env.scene.keys() else None
                command_text = (
                    command_idx.detach().cpu().tolist() if command_idx is not None else None
                )
                joint_text = (
                    tracked.data.joint_pos.detach().cpu().tolist()
                    if tracked is not None and hasattr(tracked.data, "joint_pos")
                    else None
                )
                screw_target = getattr(command_term, "_screw_revolute_target", None)
                screw_target_text = (
                    screw_target.detach().cpu().tolist()
                    if screw_target is not None
                    else None
                )
                print(
                    f"[POSE] step={count} command_idx={command_text} "
                    f"joint_pos={joint_text} "
                    f"screw_target={screw_target_text}",
                    flush=True,
                )
                if command_idx is not None:
                    active_idx = int(command_idx[debug_env].item())
                    handlers = getattr(command_term, "_command_handlers", ())
                    if active_idx < len(handlers):
                        env_ids = torch.tensor(
                            [debug_env], device=base_env.device, dtype=torch.long
                        )
                        ee_pos_b, ee_quat_b = command_term._get_ee_in_base_frame(env_ids)
                        target_pos_b, target_quat_b = handlers[
                            active_idx
                        ].get_target_in_base_frame(env_ids)
                        pos_err = torch.linalg.vector_norm(ee_pos_b - target_pos_b, dim=-1)
                        ori_err = math_utils.quat_error_magnitude(ee_quat_b, target_quat_b)
                        print(
                            f"[POSE] env={debug_env} ee_pos_b={ee_pos_b.cpu().tolist()} "
                            f"target_pos_b={target_pos_b.cpu().tolist()} "
                            f"pos_err={pos_err.cpu().tolist()} "
                            f"ori_err_deg={torch.rad2deg(ori_err).cpu().tolist()}",
                            flush=True,
                        )

            term = terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)
            trunc = truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
            if args_cli.video and (term or trunc or count >= record_steps):
                break
            if args_cli.max_steps is not None and count >= args_cli.max_steps:
                break

    env.close()

    if args_cli.lamp_distance_plot is not None and lamp_distance_rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = Path(args_cli.lamp_distance_plot).expanduser().resolve()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = plot_path.with_suffix(".csv")
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=lamp_distance_rows[0].keys())
            writer.writeheader()
            writer.writerows(lamp_distance_rows)

        times = [row["time_s"] for row in lamp_distance_rows]
        screw_mask = [row["command_type"] == "_ScrewFrameHandler" for row in lamp_distance_rows]
        fig, (distance_ax, insertion_ax) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
        )
        distance_ax.plot(
            times,
            [row["distance_mm"] for row in lamp_distance_rows],
            label="Total distance",
            linewidth=2.0,
        )
        distance_ax.plot(
            times,
            [row["radial_mm"] for row in lamp_distance_rows],
            label="Radial offset",
        )
        distance_ax.plot(
            times,
            [row["axial_mm"] for row in lamp_distance_rows],
            label="Signed axial offset",
        )
        screw_times = [time for time, active in zip(times, screw_mask) if active]
        screw_distances = [
            row["distance_mm"]
            for row, active in zip(lamp_distance_rows, screw_mask)
            if active
        ]
        distance_ax.scatter(
            screw_times,
            screw_distances,
            s=8,
            color="tab:orange",
            label="Screw stage",
            zorder=3,
        )
        distance_ax.axhline(0.0, color="black", linewidth=0.7)
        distance_ax.set_ylabel("TCP relative to bulb [mm]")
        distance_ax.grid(alpha=0.25)
        distance_ax.legend(ncols=2)

        insertion_ax.plot(
            times,
            [row["lamp_insertion_mm"] for row in lamp_distance_rows],
            color="tab:green",
            linewidth=2.0,
        )
        insertion_ax.set_xlabel("Simulation time [s]")
        insertion_ax.set_ylabel("Prismatic position [mm]")
        insertion_ax.grid(alpha=0.25)
        fig.suptitle("Spot lamp grasp alignment during screw motion")
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"[INFO] Lamp distance trace: {csv_path}")
        print(f"[INFO] Lamp distance plot: {plot_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
