# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Valve-only joint sweep: drive RevoluteJoint open (-pi/2) -> close (0).

Pure Newton (no IsaacLab scene, no robot): parses a valve USD asset,
servo-drives the joint along a smooth minimum-jerk profile and reports
tracking error, jitter and limit behavior. Used to isolate valve physics
from robot grasp/reset issues.

Usage::

    python scripts/valve_joint_sweep.py --usd <Ball_Valve...newton.usda>
    python scripts/valve_joint_sweep.py --usd ... --seconds 5 --close-to-open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_USD = (
    REPO_ROOT
    / "source/isaaclab_hiveboard/isaaclab_hiveboard/assets/hiveboard/ball_valve/usd"
)
DEFAULT_USD = EXT_USD / "Ball_Valve_uuc_newton.usda"

Q_OPEN, Q_CLOSED = -1.5707963267948966, 0.0


def put(wp_arr, values):
    """Write host values into a (possibly CUDA) warp array in place."""
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    host = np.asarray(values, dtype=np.float32).reshape(wp_arr.shape)
    wp_arr.assign(wp.array(host, dtype=wp_arr.dtype, device=wp_arr.device))


def profile(t: float, dur: float) -> float:
    """Minimum-jerk 0->1 over ``dur``."""
    s = min(max(t / dur, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def main(argv: list[str] | None = None) -> int:
    import numpy as np  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", default=str(DEFAULT_USD))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--stiffness", type=float, default=200.0)
    parser.add_argument("--damping", type=float, default=20.0)
    parser.add_argument(
        "--close-to-open",
        action="store_true",
        help="Sweep close -> open instead of open -> close.",
    )
    args = parser.parse_args(argv)

    q_from, q_to = (Q_CLOSED, Q_OPEN) if args.close_to_open else (Q_OPEN, Q_CLOSED)

    builder = newton.ModelBuilder(up_axis="Z")
    builder.add_usd(args.usd)
    print("bodies:", list(builder.body_label))
    print("joints:", list(builder.joint_label), [int(t) for t in builder.joint_type])

    model = builder.finalize(device=args.device)
    model.set_gravity((0.0, 0.0, 0.0))  # task disables gravity on the valve
    put(model.joint_target_ke, [args.stiffness])
    put(model.joint_target_kd, [args.damping])
    put(model.joint_target_mode, [int(newton.JointTargetMode.POSITION)])

    state0 = model.state()
    put(state0.joint_q, [q_from])
    put(state0.joint_qd, [0.0])
    solver = newton.solvers.SolverMuJoCo(model)
    state1 = model.state()
    control = model.control()

    n = int(args.seconds / args.dt) + int(args.settle_s / args.dt)
    qs, qds, qts = [], [], []
    t = 0.0
    for _ in range(n):
        q_des = q_from + (q_to - q_from) * profile(t, args.seconds)
        put(control.joint_target_pos, [q_des])
        contacts = model.collide(state0)
        solver.step(state0, state1, control, contacts, args.dt)
        state0, state1 = state1, state0
        t += args.dt
        qs.append(float(state0.joint_q.numpy()[0]))
        qds.append(float(state0.joint_qd.numpy()[0]))
        qts.append(q_des)

    qs = np.array(qs)
    qds = np.array(qds)
    sw = qs[: int(args.seconds / args.dt)]
    print(f"sweep steps: {len(sw)}, final q: {qs[-1]:.5f} (target {q_to})")
    print(f"max |q - qdes| during sweep: {np.abs(sw - np.array(qts[: len(sw)])).max():.5f} rad")
    print(f"qd std during sweep: {qds[: len(sw)].std():.5f}")
    print(f"settle qd std: {qds[len(sw):].std():.6f}")
    print(f"q range: [{qs.min():.4f}, {qs.max():.4f}] (limits [{Q_OPEN}, {Q_CLOSED}])")
    stride = max(1, len(sw) // 10)
    for k in range(0, len(sw), stride):
        print(f"  t={k * args.dt:4.1f}s q={sw[k]:+.4f} des={qts[k]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
