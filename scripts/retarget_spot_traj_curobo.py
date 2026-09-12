#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate the bench Spot clip from traj_edit beads using cuRobo IK.

Same contract as ``retarget_spot_traj.py`` (keys in, ``q`` out) but each
Cartesian sample is solved with cuRobo's multi-seed global IK instead of
chained damped-least-squares. Consecutive samples keep the nearest branch to
the previous solution, so the clip stays continuous where DLS flipped branches
(samples ~515-517 and ~561 in the MuJoCo clip).

No simulator needed: targets come from the beads, kinematics from cuRobo's
Spot URDF. The gripper channel is snapped to open/close at -0.9, matching the
gains env's binary gripper action.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.spot.bench import (
    ARM_JOINT_NAMES,
    BENCH_JOINT_POS,
    BODY_POS,
    TCP_SITE_POS,
    TCP_SITE_QUAT_XYZW,
    TRAJECTORY_JSON,
)
from isaaclab_hiveboard.mdp.curobo_robot_cfg import load_curobo_robot_cfg
from isaaclab_hiveboard.mdp.curobo_warp import curobo_compatible_warp
from isaaclab_hiveboard.utils.spot_traj import (
    expand_keys,
    load_payload,
    n_samples_from_keys,
    print_key_report,
    save_payload,
)

ARM_NAMES = list(ARM_JOINT_NAMES[:6])
LEG_LOCKS = {
    k: float(v)
    for k, v in BENCH_JOINT_POS.items()
    if k[:2] in ("fl", "fr", "hl", "hr")
}
GRIP_OPEN, GRIP_CLOSE, GRIP_THRESHOLD = -1.5, -0.3, -0.9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default=str(TRAJECTORY_JSON), help="Input trajectory JSON with keys + q.")
    parser.add_argument(
        "--out",
        default=str(Path(TRAJECTORY_JSON).with_name("spot_bench_gains_curobo.json")),
        help="Output JSON for the cuRobo-solved clip.",
    )
    parser.add_argument("--seeds", type=int, default=8, help="IK solutions kept per sample; nearest to previous wins.")
    parser.add_argument("--chunk", type=int, default=48, help="Samples solved per batched cuRobo call.")
    parser.add_argument("--dry-run", action="store_true", help="Solve and print residuals; do not write.")
    return parser.parse_args()


def _xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def _flange_targets(pos: np.ndarray, quat_xyzw: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Bead TCP-site path (env frame) -> wr1-link targets (base frame, cuRobo wxyz)."""
    import isaaclab.utils.math as math_utils

    site_p = torch.as_tensor(pos, dtype=torch.float32)
    site_q = torch.as_tensor(quat_xyzw, dtype=torch.float32)
    off_p = torch.as_tensor(TCP_SITE_POS, dtype=torch.float32)
    off_q = torch.as_tensor(TCP_SITE_QUAT_XYZW, dtype=torch.float32)
    site_r = math_utils.matrix_from_quat(site_q)
    off_r = math_utils.matrix_from_quat(off_q)
    flange_r = site_r @ off_r.transpose(-1, -2)
    flange_p = site_p - torch.bmm(flange_r, off_p.reshape(1, 3, 1).expand(len(pos), 3, 1)).squeeze(-1)
    flange_q = math_utils.quat_from_matrix(flange_r)
    root = torch.as_tensor(BODY_POS, dtype=torch.float32)
    return flange_p - root, _xyzw_to_wxyz(flange_q)


def _wrapped_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    delta = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return torch.linalg.vector_norm(delta, dim=-1)


def main() -> int:
    args = _parse_args()
    payload = load_payload(args.traj)
    if "keys" not in payload or "q" not in payload:
        raise SystemExit(f"{args.traj} needs both 'keys' and 'q'.")
    keys = payload["keys"]
    n_samples = n_samples_from_keys(keys)

    with curobo_compatible_warp():
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

        robot_cfg = load_curobo_robot_cfg(
            f"{ASSET_DIR}/spot/cumotion/spot.yaml",
            f"{ASSET_DIR}/spot/spot_with_arm.urdf",
            lock_joints=LEG_LOCKS,
        )
        with torch.inference_mode(False):
            ik = InverseKinematics(
                InverseKinematicsCfg.create(
                    robot=robot_cfg,
                    num_seeds=max(16, args.seeds * 2),
                    position_tolerance=0.005,
                    orientation_tolerance=0.05,
                    self_collision_check=False,
                    use_cuda_graph=False,
                    max_batch_size=args.chunk,
                    device_cfg=DeviceCfg(),
                )
            )
        link = ik.tool_frames[0]
        sol_names = list(ik.default_joint_state.joint_names)
        arm_idx = [sol_names.index(name) for name in ARM_NAMES]

        # Seed the chain from the input clip so the branch matches the old motion.
        q_src = np.asarray(payload["q"], dtype=np.float64)
        prev = torch.as_tensor(q_src[0, :6], dtype=torch.float32, device="cuda:0")

        # Home site orientation measured in cuRobo-land, mirroring retarget().
        home_state = JointState.from_position(
            torch.zeros(len(sol_names), dtype=torch.float32, device="cuda:0").unsqueeze(0),
            joint_names=sol_names,
        )
        home_state.position[0, arm_idx] = torch.as_tensor(
            q_src[0, :6], dtype=torch.float32, device="cuda:0"
        )
        with torch.inference_mode(False):
            home_tcp = ik.compute_kinematics(home_state).tool_poses.get_link_pose(link)
        # get_link_pose returns wxyz; expand_keys works in xyzw. Compose the
        # wr1-link orientation with the site offset: the DLS retarget measures
        # this same site frame via Isaac FK.
        import isaaclab.utils.math as math_utils

        flange_xyzw = home_tcp.quaternion[0].detach().cpu()[[1, 2, 3, 0]]
        off_xyzw = torch.as_tensor(TCP_SITE_QUAT_XYZW, dtype=torch.float32)
        home_site_q = math_utils.quat_mul(flange_xyzw, off_xyzw).numpy().astype(np.float64)

        pos, quat, grip, _ = expand_keys(keys, n_samples, home_site_q)
        goal_p, goal_q = _flange_targets(pos, quat)
        device = torch.device("cuda:0")
        goal_p, goal_q = goal_p.to(device), goal_q.to(device)

        q_arm = torch.zeros(n_samples, 6, dtype=torch.float32)
        pos_err = torch.zeros(n_samples, dtype=torch.float32)
        rot_err = torch.zeros(n_samples, dtype=torch.float32)
        failed: list[int] = []
        max_step = 0.0
        with torch.inference_mode(False):
            for start in range(0, n_samples, args.chunk):
                end = min(start + args.chunk, n_samples)
                goal = GoalToolPose.from_poses(
                    {link: Pose(position=goal_p[start:end].to(device), quaternion=goal_q[start:end].to(device))},
                    ordered_tool_frames=ik.tool_frames,
                    num_goalset=1,
                )
                res = ik.solve_pose(goal, return_seeds=args.seeds)
                sols = res.js_solution.position  # (batch, seeds, dof)
                ok = res.success  # (batch, seeds)
                for b in range(end - start):
                    dist = _wrapped_dist(sols[b, :, arm_idx], prev.unsqueeze(0).expand_as(sols[b, :, arm_idx]))
                    # Soft cost, never a hard gate: positions pin the solution
                    # (meters), joint-space distance keeps the branch so the
                    # clip cannot jump, rotation is only a gentle tiebreak.
                    # The DLS reference itself carries tens of degrees of
                    # orientation residual; position fidelity + continuity is
                    # what gains validation needs. Strict-tol misses are only
                    # reported, never held.
                    score = dist + 20.0 * res.position_error[b] + 0.2 * res.rotation_error[b]
                    best = int(torch.argmin(score))
                    if not bool(ok[b, best]):
                        failed.append(start + b)
                    prev = sols[b, best, arm_idx].detach().clone()
                    q_arm[start + b] = prev.cpu()
                    pos_err[start + b] = float(res.position_error[b, best])
                    rot_err[start + b] = float(res.rotation_error[b, best])
                    if start + b > 0:
                        step_q = q_arm[start + b - 1].to(prev.device)
                        max_step = max(max_step, float(_wrapped_dist(prev, step_q).max()))

            # Repair pass: re-solve jump samples warm-started from the previous
            # accepted solution so the optimizer refines the continuous branch
            # instead of settling for a far converged one.
            full_names = list(res.js_solution.joint_names)
            full_arm_idx = [full_names.index(name) for name in ARM_NAMES]
            lock_state = dict(LEG_LOCKS)
            lock_state["arm_f1x"] = 0.0
            JUMP_THRESHOLD = 0.15
            for _ in range(3):
                repaired_any = False
                for i in range(1, n_samples):
                    prev_q = q_arm[i - 1]
                    step_in = float(_wrapped_dist(q_arm[i], prev_q).max())
                    if step_in <= JUMP_THRESHOLD:
                        continue
                    nxt_q = q_arm[i + 1] if i + 1 < n_samples else q_arm[i]
                    step_out = float(_wrapped_dist(nxt_q, q_arm[i]).max())
                    # SeedManager ignores current_state; seed_config (batch, n,
                    # dof) is the actual warm-start input, topped up randomly.
                    seed_prev = prev_q.to(device).unsqueeze(0).unsqueeze(0).expand(1, 4, 6).clone()
                    seed_prev += 0.02 * torch.randn_like(seed_prev)
                    goal_one = GoalToolPose.from_poses(
                        {link: Pose(position=goal_p[i : i + 1], quaternion=goal_q[i : i + 1])},
                        ordered_tool_frames=ik.tool_frames,
                        num_goalset=1,
                    )
                    res1 = ik.solve_pose(
                        goal_one,
                        seed_config=seed_prev,
                        return_seeds=args.seeds,
                    )
                    s1 = res1.js_solution.position[0, :, full_arm_idx]
                    d1 = _wrapped_dist(s1, prev_q.to(device).unsqueeze(0).expand_as(s1))
                    sc = d1 + 20.0 * res1.position_error[0] + 0.2 * res1.rotation_error[0]
                    cand = s1[int(torch.argmin(sc))].detach().cpu()
                    new_in = float(_wrapped_dist(cand, prev_q).max())
                    new_out = float(_wrapped_dist(nxt_q, cand).max())
                    if max(new_in, new_out) < max(step_in, step_out):
                        q_arm[i] = cand
                        pos_err[i] = float(res1.position_error[0, int(torch.argmin(sc))])
                        rot_err[i] = float(res1.rotation_error[0, int(torch.argmin(sc))])
                        repaired_any = True
                if not repaired_any:
                    break

        q_out = np.zeros((n_samples, 7), dtype=np.float64)
        q_out[:, :6] = q_arm.numpy()
        q_out[:, 6] = np.where(np.asarray(grip) > GRIP_THRESHOLD, GRIP_CLOSE, GRIP_OPEN)

        key_report = []
        for key in keys:
            key_report.append({
                "id": key.get("id"),
                "label": key.get("label"),
                "sample": min(int(key["sample"]), n_samples - 1),
                "err_xyz_mm": [0.0, 0.0, 0.0],
                "err_norm_mm": 0.0,
                "err_rot_deg": 0.0,
            })
        # cuRobo FK residual vs the flange targets (self-consistency of the
        # solve). NOTE: comparing against the beads here would just reproduce
        # the ~200 mm site offset; the flange rides behind the TCP by design.
        with torch.inference_mode(False):
            full = torch.zeros(len(keys), len(sol_names), dtype=torch.float32, device=device)
            arm_q = torch.as_tensor(q_out[[min(int(k["sample"]), n_samples - 1) for k in keys], :6])
            for j, name in enumerate(ARM_NAMES):
                full[:, sol_names.index(name)] = arm_q[:, j]
            states = JointState.from_position(full, joint_names=sol_names)
            fk = ik.compute_kinematics(states).tool_poses.get_link_pose(link)
        fk_p = fk.position.detach().float().cpu().numpy()
        for row in key_report:
            sample = row["sample"]
            target = (goal_p[sample].detach().float().cpu().numpy())
            got = fk_p[key_report.index(row)]
            row["err_xyz_mm"] = [round(float(v) * 1000.0, 2) for v in got - target]
            row["err_norm_mm"] = round(float(np.linalg.norm(got - target) * 1000.0), 2)

    step_mag = torch.zeros(n_samples, dtype=torch.float32)
    for i in range(1, n_samples):
        step_mag[i] = float(_wrapped_dist(q_arm[i], q_arm[i - 1]).max())
    top = torch.topk(step_mag, k=min(5, n_samples))
    print(f"[INFO] solved {n_samples} samples, {len(failed)} strict-tol misses (tracked anyway)")
    print("[INFO] top per-sample joint steps (sample: rad): " + ", ".join(
        f"{int(i)}: {float(v):.3f}" for v, i in zip(top.values, top.indices)
    ))
    if failed:
        ranges: list[str] = []
        lo = prev_lo = failed[0]
        for s in failed[1:]:
            if s == prev_lo + 1:
                prev_lo = s
                continue
            ranges.append(f"{lo}-{prev_lo}" if prev_lo > lo else f"{lo}")
            lo = prev_lo = s
        ranges.append(f"{lo}-{prev_lo}" if prev_lo > lo else f"{lo}")
        print(f"[INFO] failed samples: {', '.join(ranges)}")
    print(f"[INFO] max per-sample joint step: {max_step:.4f} rad")
    print(f"[INFO] mean cuRobo position residual: {float(pos_err.mean()):.5f} m")
    print(f"[INFO] max cuRobo rotation residual: {float(rot_err.max()):.4f} rad")
    print_key_report(key_report)
    if args.dry_run:
        return 0

    payload["q"] = np.round(q_out, 6).tolist()
    payload["source"] = f"retargeted from {Path(args.traj).name} with cuRobo chained global IK"
    payload["keys_source"] = "traj_edit beads; cuRobo IK on website tcp site, nearest-branch chaining"
    save_payload(args.out, payload)
    print(f"[INFO] Wrote {len(q_out)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
