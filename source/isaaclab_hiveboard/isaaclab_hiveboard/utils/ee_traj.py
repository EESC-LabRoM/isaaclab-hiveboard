# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump end-effector / command trajectories from a HiveBoard play episode."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import RotateFrameCfg


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


class EeTrajDumper:
    """Record TCP, command, rotation-axis, and valve joint traces for plotting."""

    def __init__(self, env, out_dir: str, env_index: int = 0):
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._out_dir = os.path.abspath(out_dir)
        self._i = int(env_index)
        self._rows: list[dict[str, Any]] = []
        os.makedirs(self._out_dir, exist_ok=True)

        self._robot = self._env.scene["robot"]
        self._cmd = self._env.command_manager.get_term("pose_command")
        self._frame = self._env.scene["target_frame"]
        self._frame_names = list(self._frame.data.target_frame_names)
        self._cmd_names = [type(cfg).__name__.removesuffix("Cfg") for cfg in self._cmd.cfg.commands]
        self._rotate_cfg = next(
            (cfg for cfg in self._cmd.cfg.commands if isinstance(cfg, RotateFrameCfg)),
            None,
        )
        self._axis_local = None
        if self._rotate_cfg is not None:
            self._axis_local = torch.tensor(
                self._rotate_cfg.axis,
                device=self._env.device,
                dtype=torch.float32,
            )

        self._valve = None
        self._valve_joint_idx = None
        for name in ("high_torque_valve", "small_valve", "ball_valve", "lever_valve"):
            if name in self._env.scene.keys():
                self._valve = self._env.scene[name]
                joint_ids, _ = self._valve.find_joints("RevoluteJoint")
                if joint_ids:
                    self._valve_joint_idx = joint_ids[0]
                break

        wr1_ids, _ = self._robot.find_bodies("arm_link_wr1")
        self._wr1_idx = wr1_ids[0] if wr1_ids else self._cmd._body_idx

    def sample(
        self,
        step: int,
        applied_command: torch.Tensor | None = None,
        command_idx: int | None = None,
    ) -> None:
        i = self._i
        device = self._env.device
        env_ids = torch.tensor([i], device=device, dtype=torch.long)

        ee_pos_b, ee_quat_b = self._cmd._get_ee_in_base_frame(env_ids)
        ee_pos_w, ee_quat_w = self._cmd._get_ee_in_world_frame(env_ids)
        command = applied_command[i] if applied_command is not None else self._cmd.command[i]
        cmd_pos_b = command[1:4]
        cmd_quat_b = command[4:8]
        cmd_idx = (
            int(command_idx)
            if command_idx is not None
            else int(self._cmd._current_command_idx[i].item())
        )
        cmd_name = self._cmd_names[cmd_idx] if cmd_idx < len(self._cmd_names) else "done"

        wr1_pos_w = self._robot.data.body_pos_w[i, self._wr1_idx]
        wr1_quat_w = self._robot.data.body_quat_w[i, self._wr1_idx]
        wr1_pos_b, wr1_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w[i : i + 1],
            self._robot.data.root_quat_w[i : i + 1],
            wr1_pos_w.unsqueeze(0),
            wr1_quat_w.unsqueeze(0),
        )

        frame_pos_w = self._frame.data.target_pos_w[i]
        frame_quat_w = self._frame.data.target_quat_w[i]
        n_frames = frame_pos_w.shape[0]
        root_pos = self._robot.data.root_pos_w[i].unsqueeze(0).expand(n_frames, -1)
        root_quat = self._robot.data.root_quat_w[i].unsqueeze(0).expand(n_frames, -1)
        frame_pos_b, frame_quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, frame_pos_w, frame_quat_w
        )

        rotate_idx = (
            self._frame_names.index(self._rotate_cfg.target_frame_name)
            if self._rotate_cfg is not None and self._rotate_cfg.target_frame_name in self._frame_names
            else 0
        )
        axis_pos_b = frame_pos_b[rotate_idx]
        axis_quat_b = frame_quat_b[rotate_idx]
        if self._axis_local is not None:
            rot_axis_b = math_utils.quat_apply(axis_quat_b.unsqueeze(0), self._axis_local.unsqueeze(0))[0]
            rot_axis_b = rot_axis_b / torch.linalg.vector_norm(rot_axis_b).clamp(min=1.0e-8)
        else:
            rot_axis_b = torch.tensor([1.0, 0.0, 0.0], device=device)

        radius_vec = ee_pos_b[0] - axis_pos_b
        axial = torch.dot(radius_vec, rot_axis_b)
        radial = radius_vec - axial * rot_axis_b
        radius = torch.linalg.vector_norm(radial)
        pos_err = torch.linalg.vector_norm(ee_pos_b[0] - cmd_pos_b)
        ori_err = math_utils.quat_error_magnitude(ee_quat_b[0].unsqueeze(0), cmd_quat_b.unsqueeze(0))[0]

        valve_q = 0.0
        if self._valve is not None and self._valve_joint_idx is not None:
            valve_q = float(self._valve.data.joint_pos[i, self._valve_joint_idx].item())

        row: dict[str, Any] = {
            "step": step,
            "time_s": step * float(self._env.step_dt),
            "command_idx": cmd_idx,
            "command_name": cmd_name,
            "gripper_open": float(command[0].item()),
            "valve_joint_rad": valve_q,
            "radius_m": float(radius.item()),
            "axial_m": float(axial.item()),
            "pos_err_m": float(pos_err.item()),
            "ori_err_rad": float(ori_err.item()),
            "ee_x_b": float(ee_pos_b[0, 0]),
            "ee_y_b": float(ee_pos_b[0, 1]),
            "ee_z_b": float(ee_pos_b[0, 2]),
            "ee_qw_b": float(ee_quat_b[0, 0]),
            "ee_qx_b": float(ee_quat_b[0, 1]),
            "ee_qy_b": float(ee_quat_b[0, 2]),
            "ee_qz_b": float(ee_quat_b[0, 3]),
            "cmd_x_b": float(cmd_pos_b[0]),
            "cmd_y_b": float(cmd_pos_b[1]),
            "cmd_z_b": float(cmd_pos_b[2]),
            "cmd_qw_b": float(cmd_quat_b[0]),
            "cmd_qx_b": float(cmd_quat_b[1]),
            "cmd_qy_b": float(cmd_quat_b[2]),
            "cmd_qz_b": float(cmd_quat_b[3]),
            "wr1_x_b": float(wr1_pos_b[0, 0]),
            "wr1_y_b": float(wr1_pos_b[0, 1]),
            "wr1_z_b": float(wr1_pos_b[0, 2]),
            "axis_x_b": float(axis_pos_b[0]),
            "axis_y_b": float(axis_pos_b[1]),
            "axis_z_b": float(axis_pos_b[2]),
            "rot_axis_x_b": float(rot_axis_b[0]),
            "rot_axis_y_b": float(rot_axis_b[1]),
            "rot_axis_z_b": float(rot_axis_b[2]),
            "ee_x_w": float(ee_pos_w[0, 0]),
            "ee_y_w": float(ee_pos_w[0, 1]),
            "ee_z_w": float(ee_pos_w[0, 2]),
        }
        for name, pos in zip(self._frame_names, frame_pos_b):
            row[f"{name}_x_b"] = float(pos[0])
            row[f"{name}_y_b"] = float(pos[1])
            row[f"{name}_z_b"] = float(pos[2])
        self._rows.append(row)

    def save(self) -> str:
        if not self._rows:
            raise RuntimeError("EeTrajDumper.save() called with no samples.")

        keys = list(self._rows[0].keys())
        csv_path = os.path.join(self._out_dir, "ee_traj.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._rows)

        arrays: dict[str, np.ndarray] = {}
        meta = {"command_names": self._cmd_names, "frame_names": self._frame_names, "n_steps": len(self._rows)}
        for key in keys:
            if key == "command_name":
                arrays[key] = np.array([row[key] for row in self._rows], dtype=object)
            else:
                arrays[key] = np.asarray([row[key] for row in self._rows], dtype=np.float64)
        npz_path = os.path.join(self._out_dir, "ee_traj.npz")
        np.savez_compressed(npz_path, **arrays)

        meta_path = os.path.join(self._out_dir, "ee_traj_meta.json")
        with open(meta_path, "w") as handle:
            json.dump(meta, handle, indent=2)

        plot_paths = _plot_traj(arrays, self._out_dir)
        print(f"[INFO] Wrote EE trajectory dataset: {csv_path}")
        print(f"[INFO] NPZ: {npz_path}")
        for path in plot_paths:
            print(f"[INFO] Plot: {path}")
        return csv_path


def _plot_traj(data: dict[str, np.ndarray], out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    t = data["time_s"]
    cmd_idx = data["command_idx"]
    rotate_mask = np.array(["Rotate" in str(name) for name in data["command_name"]])

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.plot(data["ee_y_b"], data["ee_z_b"], color="C0", label="TCP")
    ax.plot(data["cmd_y_b"], data["cmd_z_b"], color="C1", linestyle="--", label="command")
    if rotate_mask.any():
        ax.plot(
            data["ee_y_b"][rotate_mask],
            data["ee_z_b"][rotate_mask],
            color="C0",
            linewidth=2.5,
            label="TCP (rotate)",
        )
        ax.plot(
            data["cmd_y_b"][rotate_mask],
            data["cmd_z_b"][rotate_mask],
            color="C1",
            linewidth=2.5,
            linestyle="--",
            label="command (rotate)",
        )
    ax.scatter(data["axis_y_b"][0], data["axis_z_b"][0], c="k", marker="x", s=80, label="rotate_frame")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("y base [m]")
    ax.set_ylabel("z base [m]")
    ax.set_title("TCP in the valve rotation plane (base YZ)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "ee_path_yz.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.0), sharex=True)
    axes[0].plot(t, data["ee_x_b"], label="TCP")
    axes[0].plot(t, data["cmd_x_b"], linestyle="--", label="command")
    axes[0].plot(t, data["axis_x_b"], linestyle=":", label="rotate_frame")
    axes[0].set_ylabel("x [m]")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(t, data["ee_y_b"], label="TCP")
    axes[1].plot(t, data["cmd_y_b"], linestyle="--", label="command")
    axes[1].plot(t, data["axis_y_b"], linestyle=":", label="rotate_frame")
    axes[1].set_ylabel("y [m]")
    axes[2].plot(t, data["ee_z_b"], label="TCP")
    axes[2].plot(t, data["cmd_z_b"], linestyle="--", label="command")
    axes[2].plot(t, data["axis_z_b"], linestyle=":", label="rotate_frame")
    axes[2].set_ylabel("z [m]")
    axes[3].plot(t, data["radius_m"], label="TCP radius to axis")
    axes[3].plot(t, data["pos_err_m"], label="TCP vs command")
    axes[3].set_ylabel("[m]")
    axes[3].set_xlabel("time [s]")
    axes[3].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.3)
        _shade_commands(ax, t, cmd_idx)
    axes[0].set_title("TCP vs commanded pose (robot base)")
    path = os.path.join(out_dir, "ee_timeseries.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True)
    axes[0].plot(t, np.rad2deg(data["valve_joint_rad"]))
    axes[0].set_ylabel("valve joint [deg]")
    axes[1].plot(t, np.rad2deg(data["ori_err_rad"]))
    axes[1].set_ylabel("ori err [deg]")
    axes[2].plot(t, cmd_idx, drawstyle="steps-post")
    axes[2].set_ylabel("command idx")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        _shade_commands(ax, t, cmd_idx)
    axes[0].set_title("Valve joint, orientation tracking, command phase")
    path = os.path.join(out_dir, "valve_and_phases.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data["ee_x_b"], data["ee_y_b"], data["ee_z_b"], color="C0", label="TCP")
    ax.plot(data["cmd_x_b"], data["cmd_y_b"], data["cmd_z_b"], color="C1", linestyle="--", label="command")
    ax.scatter(
        data["axis_x_b"][0],
        data["axis_y_b"][0],
        data["axis_z_b"][0],
        c="k",
        marker="x",
        s=60,
        label="rotate_frame",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("TCP path in robot base")
    ax.legend(loc="best", fontsize=8)
    path = os.path.join(out_dir, "ee_path_3d.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def _shade_commands(ax, t: np.ndarray, cmd_idx: np.ndarray) -> None:
    if t.size == 0:
        return
    changes = np.flatnonzero(np.diff(cmd_idx) != 0) + 1
    bounds = np.concatenate(([0], changes, [t.size - 1]))
    colors = ["#dddddd", "#eeeeee"]
    for i in range(len(bounds) - 1):
        ax.axvspan(t[bounds[i]], t[bounds[i + 1]], color=colors[i % 2], lw=0)
