# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dump commanded vs measured arm joints from a HiveBoard playback episode."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np
import torch


CONTACT_FORCE_THRESHOLD_N = 1.0


def _as_numpy(value) -> np.ndarray:
    tensor = value.torch if hasattr(value, "torch") else value
    return tensor.detach().float().cpu().numpy()


def _force_norm(value) -> float:
    if value is None:
        return float("nan")
    array = _as_numpy(value).reshape(-1, 3)
    return float(np.linalg.norm(array, axis=-1).max())


class JointTrajDumper:
    """Record applied joint targets, measured joints, and per-joint error."""

    def __init__(self, env, out_dir: str, env_index: int = 0):
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._out_dir = os.path.abspath(out_dir)
        self._i = int(env_index)
        self._rows: list[dict[str, Any]] = []
        os.makedirs(self._out_dir, exist_ok=True)

        self._robot = self._env.scene["robot"]
        self._dt = float(self._env.cfg.sim.dt) * float(self._env.cfg.decimation)
        self._action = self._env.action_manager.get_term("arm_action")
        self._joint_ids = self._action._joint_ids
        self._joint_names = list(self._action._joint_names)
        self._contact_names = [
            name for name in self._env.scene.keys() if str(name).endswith("_contact")
        ]

        self._valve = None
        self._valve_joint_idx = None
        if "ball_valve" in self._env.scene.keys():
            self._valve = self._env.scene["ball_valve"]
            joint_ids, _ = self._valve.find_joints("RevoluteJoint")
            if joint_ids:
                self._valve_joint_idx = joint_ids[0]

    def sample(self, step: int, applied_command: torch.Tensor) -> None:
        i = self._i
        q_cmd = _as_numpy(applied_command[i])
        q_act = _as_numpy(self._robot.data.joint_pos)[i, self._joint_ids]
        q_vel = _as_numpy(self._robot.data.joint_vel)[i, self._joint_ids]
        q_err = q_act - q_cmd

        row: dict[str, Any] = {
            "step": int(step),
            "time_s": float(step) * self._dt,
        }
        for name, cmd, act, vel, err in zip(
            self._joint_names, q_cmd, q_act, q_vel, q_err, strict=True
        ):
            row[f"{name}_cmd"] = float(cmd)
            row[f"{name}_act"] = float(act)
            row[f"{name}_vel"] = float(vel)
            row[f"{name}_err"] = float(err)
        if self._valve is not None and self._valve_joint_idx is not None:
            row["valve_q"] = float(_as_numpy(self._valve.data.joint_pos)[i, self._valve_joint_idx])
            row["valve_qd"] = float(_as_numpy(self._valve.data.joint_vel)[i, self._valve_joint_idx])
        for sensor_name in self._contact_names:
            sensor = self._env.scene[sensor_name]
            net = _force_norm(sensor.data.net_forces_w[i] if sensor.data.net_forces_w is not None else None)
            matrix = sensor.data.force_matrix_w
            valve_n = _force_norm(None if matrix is None else matrix[i])
            hitting = int(valve_n >= CONTACT_FORCE_THRESHOLD_N) if np.isfinite(valve_n) else 0
            row[f"{sensor_name}_net_N"] = net
            row[f"{sensor_name}_valve_N"] = valve_n
            row[f"{sensor_name}_hit"] = hitting
        self._rows.append(row)

    def save(self) -> str:
        if not self._rows:
            raise RuntimeError("JointTrajDumper.save() called with no samples.")

        keys = list(self._rows[0].keys())
        csv_path = os.path.join(self._out_dir, "joint_traj.csv")
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._rows)

        arrays = {key: np.asarray([row[key] for row in self._rows], dtype=np.float64) for key in keys}
        np.savez_compressed(os.path.join(self._out_dir, "joint_traj.npz"), **arrays)

        summary = self._summary(arrays)
        key_errors = self._key_errors()
        if key_errors:
            summary["key_errors"] = key_errors
            key_path = os.path.join(self._out_dir, "key_errors.json")
            with open(key_path, "w") as handle:
                json.dump(key_errors, handle, indent=2)
            print(f"[INFO] Keyframe TCP errors: {key_path}")
        with open(os.path.join(self._out_dir, "joint_traj_summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2)

        plot_paths = _plot_joint_traj(arrays, self._joint_names, self._out_dir)
        print(f"[INFO] Wrote joint tracking CSV: {csv_path}")
        print(f"[INFO] Summary: {os.path.join(self._out_dir, 'joint_traj_summary.json')}")
        for path in plot_paths:
            print(f"[INFO] Plot: {path}")
        for name in self._joint_names:
            stats = summary["joints"][name]
            print(
                f"[INFO] {name}: rms_err={stats['rms_rad']:.4f} rad "
                f"max_abs_err={stats['max_abs_rad']:.4f} rad "
                f"({stats['max_abs_deg']:.2f} deg)"
            )
        for name, stats in summary.get("contacts", {}).items():
            print(
                f"[INFO] {name}: hit={stats['hit_steps']}/{summary['n_steps']} steps "
                f"({100.0 * stats['hit_fraction']:.1f}%), "
                f"max_valve_N={stats['max_valve_N']:.2f}, "
                f"first_hit_s={stats['first_hit_s']}"
            )
        return csv_path

    def _summary(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        joints = {}
        for name in self._joint_names:
            err = arrays[f"{name}_err"]
            joints[name] = {
                "rms_rad": float(np.sqrt(np.mean(err * err))),
                "mean_abs_rad": float(np.mean(np.abs(err))),
                "max_abs_rad": float(np.max(np.abs(err))),
                "max_abs_deg": float(np.rad2deg(np.max(np.abs(err)))),
            }
        contacts = {}
        for name in self._contact_names:
            hit = arrays[f"{name}_hit"]
            force = arrays[f"{name}_valve_N"]
            hit_idx = np.flatnonzero(hit > 0.5)
            contacts[name] = {
                "hit_steps": int(hit_idx.size),
                "hit_fraction": float(hit.mean()),
                "max_valve_N": float(np.nanmax(force)),
                "first_hit_s": None if hit_idx.size == 0 else float(arrays["time_s"][hit_idx[0]]),
                "last_hit_s": None if hit_idx.size == 0 else float(arrays["time_s"][hit_idx[-1]]),
            }
        return {
            "n_steps": int(arrays["step"].shape[0]),
            "dt_s": self._dt,
            "joint_names": self._joint_names,
            "contact_names": self._contact_names,
            "contact_force_threshold_N": CONTACT_FORCE_THRESHOLD_N,
            "joints": joints,
            "contacts": contacts,
        }

    def _key_errors(self) -> list[dict[str, Any]]:
        if "joint_command" not in self._env.command_manager.active_terms:
            return []
        term = self._env.command_manager.get_term("joint_command")
        report = getattr(term, "key_error_report", None)
        if report is None:
            return []
        return report(self._i)


def _plot_joint_traj(
    data: dict[str, np.ndarray], joint_names: list[str], out_dir: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = data["time_s"]
    n = len(joint_names)
    paths: list[str] = []

    fig, axes = plt.subplots(n, 1, figsize=(10.0, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, joint_names, strict=True):
        ax.plot(t, data[f"{name}_cmd"], color="C1", linestyle="--", label="command")
        ax.plot(t, data[f"{name}_act"], color="C0", label="measured")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Commanded vs measured joint position")
    axes[-1].set_xlabel("time [s]")
    path = os.path.join(out_dir, "joint_cmd_vs_act.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(n, 1, figsize=(10.0, 1.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, joint_names, strict=True):
        err_deg = np.rad2deg(data[f"{name}_err"])
        ax.plot(t, err_deg, color="C3")
        ax.axhline(0.0, color="k", linewidth=0.6)
        ax.set_ylabel(f"{name}\n[deg]")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Joint tracking error (measured − command)")
    axes[-1].set_xlabel("time [s]")
    path = os.path.join(out_dir, "joint_error.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    if "valve_q" in data:
        fig, ax = plt.subplots(figsize=(10.0, 3.2))
        ax.plot(t, np.rad2deg(data["valve_q"]), color="C2")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("valve [deg]")
        ax.set_title("Ball valve joint")
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, "valve_joint.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    contact_names = [
        key[: -len("_valve_N")] for key in data if key.endswith("_valve_N")
    ]
    if contact_names:
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.4), sharex=True)
        for name in contact_names:
            axes[0].plot(t, data[f"{name}_valve_N"], label=name.replace("_contact", ""))
        axes[0].axhline(CONTACT_FORCE_THRESHOLD_N, color="k", linewidth=0.6, linestyle=":")
        axes[0].set_ylabel("valve force [N]")
        axes[0].set_title("Robot–valve contact force")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8, ncol=2)
        for index, name in enumerate(contact_names):
            axes[1].step(
                t,
                data[f"{name}_hit"] * (index + 1),
                where="post",
                label=name.replace("_contact", ""),
            )
        axes[1].set_yticks(list(range(1, len(contact_names) + 1)))
        axes[1].set_yticklabels(contact_names)
        axes[1].set_ylabel("in contact")
        axes[1].set_xlabel("time [s]")
        axes[1].grid(True, alpha=0.3)
        path = os.path.join(out_dir, "valve_contacts.png")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    return paths
