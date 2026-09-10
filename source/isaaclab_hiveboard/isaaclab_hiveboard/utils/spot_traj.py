# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartesian key expansion and DLS IK for the Spot bench-valve clip."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from isaaclab_hiveboard.assets.spot.bench import (
    ARM_JOINT_NAMES,
    BOARD_INTO,
    TCP_SITE_POS,
    TCP_SITE_QUAT_XYZW,
    TRAJ_RATE_HZ,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
    VALVE_POS,
)

IK_ITERS = 80
POS_TOL = 2e-4
ROT_TOL = 2e-3
LINK_POS_EPS = 5e-3


def _as_torch(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def load_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def n_samples_from_keys(keys: list[dict]) -> int:
    if not keys:
        return 0
    return int(keys[-1]["sample"]) + 1


def key_duration_s(keys: list[dict], index: int) -> float:
    if index <= 0:
        return 0.0
    return (int(keys[index]["sample"]) - int(keys[index - 1]["sample"])) / float(TRAJ_RATE_HZ)


def set_key_duration_s(keys: list[dict], index: int, secs: float) -> None:
    """Shift this key and every later sample so the segment length matches ``secs``."""
    if index <= 0:
        return
    new_span = max(int(round(float(secs) * TRAJ_RATE_HZ)), 1)
    old_sample = int(keys[index]["sample"])
    new_sample = int(keys[index - 1]["sample"]) + new_span
    delta = new_sample - old_sample
    if delta == 0:
        return
    for key in keys[index:]:
        key["sample"] = int(key["sample"]) + delta


def coincident_indices(keys: list[dict], index: int) -> list[int]:
    """Keys that sit on the same bead as ``index`` (pregrasp/grasp, hold/release)."""
    pos = np.asarray(keys[index]["pos"], dtype=np.float64)
    hits = []
    for i, key in enumerate(keys):
        if key.get("pos") is None:
            continue
        other = np.asarray(key["pos"], dtype=np.float64)
        if float(np.linalg.norm(pos - other)) < LINK_POS_EPS:
            hits.append(i)
    return hits or [index]


def nudge_key(keys: list[dict], index: int, delta: np.ndarray, link: bool = True) -> None:
    if keys[index].get("home"):
        return
    targets = coincident_indices(keys, index) if link else [index]
    delta = np.asarray(delta, dtype=np.float64)
    for i in targets:
        if keys[i].get("home"):
            continue
        pos = np.asarray(keys[i]["pos"], dtype=np.float64) + delta
        keys[i]["pos"] = [round(float(v), 5) for v in pos]


def insert_key_after(keys: list[dict], index: int) -> int:
    prev = keys[index]
    nxt = keys[index + 1] if index + 1 < len(keys) else None
    prev_sample = int(prev["sample"])
    if nxt is not None and nxt.get("pos") and prev.get("pos"):
        pos = 0.5 * (np.asarray(prev["pos"], dtype=np.float64) + np.asarray(nxt["pos"], dtype=np.float64))
        nxt_sample = int(nxt["sample"])
        if nxt_sample > prev_sample + 1:
            sample = int(round(0.5 * (prev_sample + nxt_sample)))
        else:
            sample = prev_sample + 1
            for later in keys[index + 1 :]:
                later["sample"] = int(later["sample"]) + 1
        grip = 0.5 * (float(prev["grip"]) + float(nxt["grip"]))
    else:
        pos = np.asarray(prev["pos"], dtype=np.float64)
        sample = prev_sample + max(int(TRAJ_RATE_HZ), 1)
        grip = float(prev["grip"])
        for later in keys[index + 1 :]:
            later["sample"] = int(later["sample"]) + max(int(TRAJ_RATE_HZ), 1)
    added = [k for k in keys if str(k.get("label", "")).startswith("added") or str(k.get("id", "")).startswith("a")]
    new_id = f"a{len(added) + 1}"
    keys.insert(
        index + 1,
        {
            "id": new_id,
            "label": "added",
            "sample": int(sample),
            "kind": "pose",
            "home": False,
            "transit": True,
            "pos": [round(float(v), 5) for v in pos],
            "grip": round(float(grip), 4),
        },
    )
    return index + 1


def remove_key(keys: list[dict], index: int) -> int:
    if len(keys) <= 2 or index <= 0 or index >= len(keys) - 1:
        return index
    if keys[index].get("home"):
        return index
    del keys[index]
    return min(index, len(keys) - 1)


def valve_q_for_sample(keys: list[dict], sample: int) -> float:
    grasp = next((k for k in keys if k.get("label") == "grasp"), None)
    arc = next((k for k in keys if k.get("kind") == "arc"), None)
    if grasp is None or arc is None:
        return float(VALVE_JOINT_CLOSED)
    g = int(grasp["sample"])
    a = int(arc["sample"])
    if sample <= g:
        return float(VALVE_JOINT_CLOSED)
    if sample >= a or a <= g:
        return float(VALVE_JOINT_OPEN)
    t = (sample - g) / float(a - g)
    return float((1.0 - t) * VALVE_JOINT_CLOSED + t * VALVE_JOINT_OPEN)


def smoothstep(s: float) -> float:
    return s * s * (3.0 - 2.0 * s)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.asarray(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _rotate(axis: np.ndarray, angle: float) -> np.ndarray:
    k = _unit(axis)
    k_x = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k_x + (1.0 - math.cos(angle)) * (k_x @ k_x)


def _arc_frame(p0: np.ndarray, p1: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, float]:
    r0 = p0 - anchor
    r1 = p1 - anchor
    axis = np.cross(r0, r1)
    n_axis = float(np.linalg.norm(axis))
    if n_axis < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = axis / n_axis
    angle = math.atan2(n_axis, float(np.dot(r0, r1)))
    return axis, angle


def _arc_pos(p0: np.ndarray, p1: np.ndarray, s: float, anchor: np.ndarray) -> np.ndarray:
    axis, angle = _arc_frame(p0, p1, anchor)
    if abs(angle) < 1e-8:
        return (1.0 - s) * p0 + s * p1
    r = _rotate(axis, s * angle) @ (p0 - anchor)
    n0, n1 = float(np.linalg.norm(p0 - anchor)), float(np.linalg.norm(p1 - anchor))
    scale = 1.0 if n0 < 1e-9 else ((1.0 - s) * n0 + s * n1) / max(float(np.linalg.norm(r)), 1e-9)
    return anchor + r * scale


def _hand_matrix(approach: np.ndarray, finger: np.ndarray) -> np.ndarray:
    """MuJoCo tcp site: +Z approach, +Y finger opening, +X = Y × Z."""
    z = _unit(approach)
    y = _unit(np.asarray(finger, float) - z * float(np.dot(finger, z)))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    quat = math_utils.quat_from_matrix(torch.as_tensor(matrix, dtype=torch.float32).unsqueeze(0))[0]
    return quat.numpy().astype(np.float64)


def _xyzw_to_matrix(xyzw: np.ndarray) -> np.ndarray:
    return math_utils.matrix_from_quat(torch.as_tensor(xyzw, dtype=torch.float32).unsqueeze(0))[0].numpy()


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
    quat = math_utils.quat_slerp(
        torch.as_tensor(q0, dtype=torch.float32),
        torch.as_tensor(q1, dtype=torch.float32),
        float(s),
    )
    return np.asarray(quat, dtype=np.float64)


def _valve_hand_quats(keys: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Grasp / post-arc site quaternions from the valve swing."""
    grasp = next((key for key in keys if key.get("label") == "grasp"), None)
    arc = next((key for key in keys if key.get("kind") == "arc"), None)
    if grasp is None or arc is None:
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return identity, identity, np.array([1.0, 0.0, 0.0]), 0.0
    p0 = np.asarray(grasp["pos"], dtype=np.float64)
    p1 = np.asarray(arc["pos"], dtype=np.float64)
    anchor = np.asarray(VALVE_POS, dtype=np.float64)
    axis, angle = _arc_frame(p0, p1, anchor)
    radial = p0 - anchor
    finger0 = _unit(np.cross(axis, radial))
    approach0 = np.asarray(BOARD_INTO, dtype=np.float64)
    rot_grasp = _hand_matrix(approach0, finger0)
    rot_arc = _rotate(axis, angle) @ rot_grasp
    return _matrix_to_xyzw(rot_grasp), _matrix_to_xyzw(rot_arc), axis, angle


def _quat_for_label(label: str, kind: str, home: np.ndarray, grasp: np.ndarray, arc: np.ndarray, prev: np.ndarray) -> np.ndarray:
    if kind == "arc":
        return arc
    if label in {"home", "home_hold"}:
        return home
    if label in {"approach", "pregrasp", "grasp"}:
        return grasp
    if label in {"hold", "release", "retreat"}:
        return arc
    return prev


def expand_keys(
    keys: list[dict], n_samples: int, home_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """50 Hz pose path through the beads. Returns pos, quat xyzw, grip, key samples."""
    samples = [int(k["sample"]) for k in keys]
    pos = np.zeros((n_samples, 3), dtype=np.float64)
    quat = np.zeros((n_samples, 4), dtype=np.float64)
    grip = np.zeros((n_samples,), dtype=np.float64)
    quat_grasp, quat_arc, axis, angle = _valve_hand_quats(keys)
    rot_grasp = _xyzw_to_matrix(quat_grasp)
    pos[min(samples[0], n_samples - 1)] = np.asarray(keys[0]["pos"], dtype=np.float64)
    grip[min(samples[0], n_samples - 1)] = float(keys[0]["grip"])
    quat[min(samples[0], n_samples - 1)] = home_xyzw
    prev_quat = home_xyzw
    for prev, key in zip(keys[:-1], keys[1:], strict=True):
        i0, i1 = int(prev["sample"]), int(key["sample"])
        p0 = np.asarray(prev["pos"], dtype=np.float64)
        p1 = np.asarray(key["pos"], dtype=np.float64)
        g0, g1 = float(prev["grip"]), float(key["grip"])
        span = max(i1 - i0, 1)
        kind = str(key.get("kind", "pose"))
        q_start = _quat_for_label(str(prev.get("label", "")), str(prev.get("kind", "pose")), home_xyzw, quat_grasp, quat_arc, prev_quat)
        q_end = _quat_for_label(str(key.get("label", "")), kind, home_xyzw, quat_grasp, quat_arc, q_start)
        prev_quat = q_end
        for step in range(1, span + 1):
            s = smoothstep(step / span)
            idx = min(i0 + step, n_samples - 1)
            if kind == "arc":
                pos[idx] = _arc_pos(p0, p1, s, np.asarray(VALVE_POS, dtype=np.float64))
                quat[idx] = _matrix_to_xyzw(_rotate(axis, s * angle) @ rot_grasp)
            else:
                pos[idx] = (1.0 - s) * p0 + s * p1
                quat[idx] = _slerp_xyzw(q_start, q_end, s)
            grip[idx] = (1.0 - s) * g0 + s * g1
    return pos, quat, grip, samples


class TcpIk:
    """Damped-least-squares IK onto the website ``tcp`` site on ``arm_link_wr1``."""

    def __init__(self, robot, env_index: int = 0):
        self.robot = robot
        self.i = int(env_index)
        ids, _ = robot.find_joints(ARM_JOINT_NAMES[:6], preserve_order=True)
        self.arm_ids = list(ids)
        grip_ids, _ = robot.find_joints(["arm_f1x"])
        self.grip_id = int(grip_ids[0])
        body_ids, _ = robot.find_bodies("arm_link_wr1")
        self.body_idx = int(body_ids[0])
        self.jacobi_body_idx = self.body_idx - 1 if robot.is_fixed_base else self.body_idx
        self.jacobi_joint_ids = [j + int(robot.num_base_dofs) for j in self.arm_ids]
        self.offset = torch.tensor(TCP_SITE_POS, device=robot.device, dtype=torch.float32)
        self.offset_quat = torch.tensor(TCP_SITE_QUAT_XYZW, device=robot.device, dtype=torch.float32)
        self.ctrl = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.001},
            ),
            num_envs=1,
            device=str(robot.device),
        )
        limits = _as_torch(robot.data.joint_pos_limits)[self.i, self.arm_ids]
        self.lo = limits[:, 0]
        self.hi = limits[:, 1]

    def _refresh(self) -> None:
        self.robot.update(0.0)

    def tcp_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = _as_torch(self.robot.data.body_pos_w)[self.i : self.i + 1, self.body_idx]
        quat = _as_torch(self.robot.data.body_quat_w)[self.i : self.i + 1, self.body_idx]
        tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
            pos, quat, self.offset.unsqueeze(0), self.offset_quat.unsqueeze(0)
        )
        return tcp_pos[0], tcp_quat[0]

    def _jacobian_w(self) -> torch.Tensor:
        jac = _as_torch(self.robot.data.body_link_jacobian_w)[
            self.i : self.i + 1, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ].clone()
        quat = _as_torch(self.robot.data.body_quat_w)[self.i : self.i + 1, self.body_idx]
        rot = math_utils.matrix_from_quat(quat)
        offset_w = torch.bmm(rot, self.offset.view(1, 3, 1)).squeeze(-1)
        jac[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_w), jac[:, 3:, :])
        return jac

    def write_arm(self, q_arm: torch.Tensor, grip: float) -> None:
        q = _as_torch(self.robot.data.joint_pos).clone()
        qd = torch.zeros_like(q)
        q[self.i, self.arm_ids] = q_arm
        q[self.i, self.grip_id] = float(grip)
        self.robot.write_joint_state_to_sim(q, qd)
        self.robot.write_data_to_sim()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        q_seed: torch.Tensor,
        grip: float,
        iters: int,
    ) -> tuple[torch.Tensor, float, float]:
        q_arm = q_seed.clone()
        command = torch.zeros(1, 7, device=q_arm.device, dtype=torch.float32)
        command[0, :3] = torch.as_tensor(target_pos, device=q_arm.device, dtype=torch.float32)
        command[0, 3:] = torch.as_tensor(target_quat, device=q_arm.device, dtype=torch.float32)
        pos_err = float("inf")
        rot_err = float("inf")
        for _ in range(iters):
            self.write_arm(q_arm, grip)
            self._refresh()
            tcp_pos, tcp_quat = self.tcp_pose_w()
            pos_err = float(torch.linalg.vector_norm(command[0, :3] - tcp_pos))
            rot_err = float(math_utils.quat_error_magnitude(tcp_quat.unsqueeze(0), command[:, 3:])[0])
            if pos_err < POS_TOL and rot_err < ROT_TOL:
                break
            self.ctrl.set_command(command)
            jac = self._jacobian_w()
            q_arm = self.ctrl.compute(
                tcp_pos.unsqueeze(0), tcp_quat.unsqueeze(0), jac, q_arm.unsqueeze(0)
            )[0]
            q_arm = torch.clamp(q_arm, self.lo + 0.02, self.hi - 0.02)
        self.write_arm(q_arm, grip)
        self._refresh()
        return q_arm, pos_err, rot_err


def retarget(
    env,
    payload: dict,
    ik_iters: int = IK_ITERS,
    start_sample: int = 0,
) -> tuple[list[list[float]], list[dict]]:
    """Solve joint ``q`` for Cartesian keys. Re-IKs from ``start_sample`` to the end."""
    robot = env.scene["robot"]
    ik = TcpIk(robot)
    keys = payload["keys"]
    n_samples = n_samples_from_keys(keys)
    q_src = np.asarray(payload.get("q") or payload.get("qpos") or [], dtype=np.float64)
    if q_src.ndim != 2 or q_src.shape[1] < 7:
        q_src = np.zeros((n_samples, 7), dtype=np.float64)
        q_src[:, :6] = (0.0, -1.9, 2.0, 0.0, -0.6, 0.0)
        q_src[:, 6] = -1.5
    if q_src.shape[0] != n_samples:
        q_new = np.zeros((n_samples, 7), dtype=np.float64)
        last = min(q_src.shape[0], n_samples)
        q_new[:last] = q_src[:last]
        if last < n_samples:
            q_new[last:] = q_src[last - 1]
        q_src = q_new
    start_sample = int(max(0, min(start_sample, n_samples - 1)))
    seed = torch.as_tensor(q_src[start_sample, :6], device=robot.device, dtype=torch.float32)
    ik.write_arm(seed, float(q_src[start_sample, 6]))
    ik._refresh()
    home_seed = torch.as_tensor(q_src[0, :6], device=robot.device, dtype=torch.float32)
    ik.write_arm(home_seed, float(q_src[0, 6]))
    ik._refresh()
    home_xyzw = ik.tcp_pose_w()[1].detach().cpu().numpy().astype(np.float64)
    ik.write_arm(seed, float(q_src[start_sample, 6]))
    ik._refresh()
    pos, quat, grip, key_samples = expand_keys(keys, n_samples, home_xyzw)
    q_out = np.array(q_src, copy=True)
    seed = torch.as_tensor(q_out[start_sample, :6], device=robot.device, dtype=torch.float32)
    for i in range(start_sample, n_samples):
        q_arm, _pos_err, _rot_err = ik.solve(pos[i], quat[i], seed, float(grip[i]), ik_iters)
        q_out[i, :6] = q_arm.detach().cpu().numpy()
        q_out[i, 6] = float(grip[i])
        seed = q_arm
    key_report = []
    for key, sample in zip(keys, key_samples, strict=True):
        sample = min(int(sample), n_samples - 1)
        q_arm = torch.as_tensor(q_out[sample, :6], device=robot.device, dtype=torch.float32)
        ik.write_arm(q_arm, float(q_out[sample, 6]))
        ik._refresh()
        tcp, tcp_quat = ik.tcp_pose_w()
        tcp_np = tcp.detach().cpu().numpy()
        target = np.asarray(key["pos"], dtype=np.float64)
        err = tcp_np - target
        rot_err = float(
            math_utils.quat_error_magnitude(
                tcp_quat.unsqueeze(0),
                torch.as_tensor(quat[sample], device=robot.device, dtype=torch.float32).unsqueeze(0),
            )[0]
        )
        key_report.append({
            "id": key.get("id"),
            "label": key.get("label"),
            "sample": sample,
            "err_xyz_mm": [round(float(v) * 1000.0, 2) for v in err],
            "err_norm_mm": round(float(np.linalg.norm(err) * 1000.0), 2),
            "err_rot_deg": round(math.degrees(rot_err), 2),
        })
    return np.round(q_out, 6).tolist(), key_report


def print_key_report(key_report: list[dict]) -> None:
    print("[INFO] IK residual at beads (tcp - target):")
    for row in key_report:
        dx, dy, dz = row["err_xyz_mm"]
        print(
            f"[IK] {row['id']} {str(row['label']):<10} sample={row['sample']} "
            f"err_xyz_mm=[{dx:+.1f}, {dy:+.1f}, {dz:+.1f}] |err|={row['err_norm_mm']:.1f} mm "
            f"rot={row['err_rot_deg']:.1f} deg"
        )
