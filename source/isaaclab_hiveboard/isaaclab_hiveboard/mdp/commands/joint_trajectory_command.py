# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback of a fixed joint-space trajectory as a command term."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import TCP_SITE_POS


class JointTrajectoryCommand(CommandTerm):
    """Emit successive samples from a vendored ``(q_arm, grip)`` clip."""

    cfg: "JointTrajectoryCommandCfg"

    def __init__(self, cfg: "JointTrajectoryCommandCfg", env):
        self._key_hit = None
        self._skip_key_sample = False
        super().__init__(cfg, env)
        payload = _load_payload(cfg.trajectory_path)
        samples = _samples_from_payload(payload, cfg.trajectory_path, cfg.command_dim)
        self._traj = samples.to(device=self.device, dtype=torch.float32)
        self._num_samples = int(self._traj.shape[0])
        self._index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._command = torch.zeros(
            self.num_envs, cfg.command_dim, device=self.device, dtype=torch.float32
        )
        self.valve_joint_des = torch.full(
            (self.num_envs,),
            float(cfg.valve_joint_open),
            device=self.device,
            dtype=torch.float32,
        )
        self.valve_task_goal = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._robot = env.scene["robot"]
        body_ids, _ = self._robot.find_bodies(cfg.ee_body_name)
        if not body_ids:
            raise ValueError(f"End-effector body {cfg.ee_body_name!r} not found on robot.")
        self._ee_body_idx = int(body_ids[0])
        self._ee_offset = torch.tensor(cfg.ee_offset, device=self.device, dtype=torch.float32)
        self._ee_offset_quat = torch.tensor((0.0, 0.0, 0.0, 1.0), device=self.device, dtype=torch.float32)
        self._frozen = False
        self._highlight_key = None
        self._load_keys(payload)
        self._apply_index()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def is_done(self) -> torch.Tensor:
        return self._index >= (self._num_samples - 1)

    def _resample_command(self, env_ids: Sequence[int] | slice | torch.Tensor):
        env_ids = self._resolve_env_ids(env_ids)
        self._index[env_ids] = 0
        self.valve_joint_des[env_ids] = float(self.cfg.valve_joint_open)
        self.valve_task_goal[env_ids] = 1.0
        if self._key_hit is not None:
            self._key_hit[env_ids] = False
            self._key_err[env_ids] = float("nan")
            self._key_tcp[env_ids] = float("nan")
        self._skip_key_sample = True
        self._apply_index(env_ids)

    def set_frozen(self, frozen: bool) -> None:
        """When frozen, keep emitting the current sample instead of advancing."""
        self._frozen = bool(frozen)

    def set_highlight_key(self, index: int | None) -> None:
        """Force the debug bead highlight. ``None`` follows playback."""
        self._highlight_key = None if index is None else int(index)

    def seek(self, sample: int, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        env_ids = self._resolve_env_ids(slice(None) if env_ids is None else env_ids)
        sample = int(max(0, min(int(sample), self._num_samples - 1)))
        self._index[env_ids] = sample
        self._apply_index(env_ids)

    def replace_payload(self, payload: dict) -> None:
        """Swap the clip and beads. Safe to call from the viewer UI thread."""
        with torch.inference_mode(False):
            samples = _samples_from_payload(payload, self.cfg.trajectory_path, self.cfg.command_dim)
            self._traj = samples.to(device=self.device, dtype=torch.float32).contiguous()
            self._num_samples = int(self._traj.shape[0])
            self._index.clamp_(max=self._num_samples - 1)
            self._load_keys(payload)
            self._apply_index()

    @property
    def index(self) -> torch.Tensor:
        return self._index

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def _update_command(self):
        # Physics has just integrated the command at ``_index``. Sample bead
        # errors before advancing so the tick matches the authored key sample.
        # Skip the compute() that follows resample: that pose is a reset
        # teleport, not tracking (the 0.1 mm home line).
        if self._skip_key_sample:
            self._skip_key_sample = False
        elif not self._frozen:
            self._sample_key_errors()
        if self._frozen:
            self._apply_index()
            return
        # In-place so torch.inference_mode() during env.step does not replace
        # ``_index`` with an inference tensor that later blocks reset().
        self._index.add_(1)
        self._index.clamp_(max=self._num_samples - 1)
        self._apply_index()

    def _update_metrics(self):
        return

    def _apply_index(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._command[env_ids] = self._traj[self._index[env_ids]]

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | torch.Tensor) -> torch.Tensor:
        if isinstance(env_ids, slice) or env_ids is None:
            return torch.arange(self.num_envs, device=self.device)
        if not isinstance(env_ids, torch.Tensor):
            return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _load_keys(self, payload: dict):
        """Cache ``traj_edit.py`` bead positions for debug visualization."""
        keys = payload.get("keys") or []
        positions = []
        samples = []
        prototypes = []
        labels = []
        for key in keys:
            pos = key.get("pos")
            if key.get("off") or pos is None:
                continue
            positions.append([float(v) for v in pos])
            samples.append(int(key.get("sample", 0)))
            prototypes.append(0 if key.get("transit") or key.get("home") else 1)
            labels.append(str(key.get("label") or key.get("id") or len(labels)))
        self._key_labels = labels
        if not positions:
            self._key_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            self._key_sample = torch.zeros(0, device=self.device, dtype=torch.long)
            self._key_proto = torch.zeros(0, device=self.device, dtype=torch.long)
            self._key_hit = None
            self._key_err = None
            self._key_tcp = None
            return
        n_key = len(positions)
        self._key_pos = torch.tensor(positions, device=self.device, dtype=torch.float32)
        self._key_sample = torch.tensor(samples, device=self.device, dtype=torch.long)
        self._key_proto = torch.tensor(prototypes, device=self.device, dtype=torch.long)
        self._key_hit = torch.zeros(self.num_envs, n_key, device=self.device, dtype=torch.bool)
        self._key_err = torch.full(
            (self.num_envs, n_key, 3), float("nan"), device=self.device, dtype=torch.float32
        )
        self._key_tcp = torch.full(
            (self.num_envs, n_key, 3), float("nan"), device=self.device, dtype=torch.float32
        )

    def _tcp_pos_w(self) -> torch.Tensor:
        body_pos = _as_torch(self._robot.data.body_pos_w)[:, self._ee_body_idx]
        body_quat = _as_torch(self._robot.data.body_quat_w)[:, self._ee_body_idx]
        offset = self._ee_offset.unsqueeze(0).expand(self.num_envs, -1)
        offset_quat = self._ee_offset_quat.unsqueeze(0).expand(self.num_envs, -1)
        tcp_pos, _ = math_utils.combine_frame_transforms(body_pos, body_quat, offset, offset_quat)
        return tcp_pos

    def _sample_key_errors(self):
        if self._key_hit is None or self._key_pos.shape[0] == 0:
            return
        if not getattr(self._robot, "is_initialized", True):
            return
        match = (self._key_sample.unsqueeze(0) == self._index.unsqueeze(1)) & ~self._key_hit
        if not bool(match.any()):
            return
        tcp = self._tcp_pos_w()
        target = self._env.scene.env_origins.unsqueeze(1) + self._key_pos.unsqueeze(0)
        err = tcp.unsqueeze(1) - target
        env_idx, key_idx = match.nonzero(as_tuple=True)
        self._key_err[env_idx, key_idx] = err[env_idx, key_idx]
        self._key_tcp[env_idx, key_idx] = tcp[env_idx]
        self._key_hit[env_idx, key_idx] = True
        if self.cfg.log_key_errors:
            for env_i, key_i in zip(env_idx.tolist(), key_idx.tolist(), strict=True):
                if env_i != 0:
                    continue
                self._print_key_error(env_i, key_i)

    def _print_key_error(self, env_index: int, key_index: int):
        err = self._key_err[env_index, key_index]
        tcp = self._key_tcp[env_index, key_index]
        target = self._env.scene.env_origins[env_index] + self._key_pos[key_index]
        err_mm = err * 1000.0
        print(
            f"[KEY] {key_index} {self._key_labels[key_index]:<10} "
            f"sample={int(self._key_sample[key_index])} "
            f"tcp=[{tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}] "
            f"target=[{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}] "
            f"err_xyz_mm=[{err_mm[0]:+.1f}, {err_mm[1]:+.1f}, {err_mm[2]:+.1f}] "
            f"|err|={float(torch.linalg.vector_norm(err_mm)):.1f} mm",
            flush=True,
        )

    def key_error_report(self, env_index: int = 0) -> list[dict]:
        """TCP minus bead position at each keyframe tick, in metres."""
        if self._key_hit is None:
            return []
        env_index = min(int(env_index), self.num_envs - 1)
        rows = []
        for key_index, label in enumerate(self._key_labels):
            hit = bool(self._key_hit[env_index, key_index])
            err = self._key_err[env_index, key_index]
            tcp = self._key_tcp[env_index, key_index]
            target = self._env.scene.env_origins[env_index] + self._key_pos[key_index]
            rows.append({
                "id": key_index,
                "label": label,
                "sample": int(self._key_sample[key_index]),
                "hit": hit,
                "target_m": [round(float(v), 5) for v in target],
                "tcp_m": None if not hit else [round(float(v), 5) for v in tcp],
                "err_xyz_m": None if not hit else [round(float(v), 5) for v in err],
                "err_xyz_mm": None if not hit else [round(float(v) * 1000.0, 2) for v in err],
                "err_norm_mm": None if not hit else round(float(torch.linalg.vector_norm(err) * 1000.0), 2),
            })
        return rows

    def print_key_errors(self, env_index: int = 0):
        rows = self.key_error_report(env_index)
        if not rows:
            return
        print("[KEY] TCP vs traj_edit beads (err = tcp - target; −z is below):", flush=True)
        for row in rows:
            if not row["hit"]:
                print(
                    f"[KEY] {row['id']} {row['label']:<10} sample={row['sample']}  not reached",
                    flush=True,
                )
                continue
            dx, dy, dz = row["err_xyz_mm"]
            print(
                f"[KEY] {row['id']} {row['label']:<10} sample={row['sample']} "
                f"err_xyz_mm=[{dx:+.1f}, {dy:+.1f}, {dz:+.1f}]  |err|={row['err_norm_mm']:.1f} mm",
                flush=True,
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "key_visualizer"):
                self.key_visualizer = VisualizationMarkers(self.cfg.key_visualizer_cfg)
            self.key_visualizer.set_visibility(True)
        elif hasattr(self, "key_visualizer"):
            self.key_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_key_pos") or self._key_pos is None or self._key_pos.shape[0] == 0:
            return
        if not hasattr(self, "key_visualizer") or not self.key_visualizer.is_visible():
            return
        origins = self._env.scene.env_origins
        pos = origins.unsqueeze(1) + self._key_pos.unsqueeze(0)
        n_env, n_key = pos.shape[:2]
        indices = self._key_proto.unsqueeze(0).expand(n_env, n_key).clone()
        highlight = self._highlight_key
        if highlight is not None and 0 <= highlight < n_key:
            indices[:, highlight] = 2
        else:
            reached = self._key_sample.unsqueeze(0) <= self._index.unsqueeze(1)
            current = reached.sum(dim=1).clamp(min=1) - 1
            indices[torch.arange(n_env, device=self.device), current] = 2
        self.key_visualizer.visualize(
            translations=pos.reshape(-1, 3),
            marker_indices=indices.reshape(-1),
        )


def _as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _load_payload(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _samples_from_payload(payload: dict, path: str | Path, command_dim: int) -> torch.Tensor:
    samples = payload.get("q", payload.get("qpos"))
    if samples is None:
        raise ValueError(f"Trajectory file {path} has no 'q' or 'qpos' array.")
    tensor = torch.tensor(samples, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != command_dim:
        raise ValueError(
            f"Trajectory {path} has shape {tuple(tensor.shape)}; expected (N, {command_dim})."
        )
    return tensor


def _key_visualizer_cfg() -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path="/Visuals/Command/traj_keys",
        markers={
            "transit": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.6, 0.85)),
            ),
            "contact": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.7, 0.15)),
            ),
            "active": sim_utils.SphereCfg(
                radius=0.024,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.95, 0.75)),
            ),
        },
    )


@configclass
class JointTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for :class:`JointTrajectoryCommand`."""

    class_type: type = JointTrajectoryCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    trajectory_path: str = ""
    command_dim: int = 7
    valve_asset_name: str | None = "ball_valve"
    valve_joint_name: str = "RevoluteJoint"
    valve_joint_closed: float = 0.0
    valve_joint_open: float = -1.5707963267948966
    debug_vis: bool = False
    key_visualizer_cfg: VisualizationMarkersCfg = _key_visualizer_cfg()
    ee_body_name: str = "arm_link_wr1"
    ee_offset: tuple[float, float, float] = TCP_SITE_POS
    log_key_errors: bool = True
