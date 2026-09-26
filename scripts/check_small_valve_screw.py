# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The small-valve stem may advance only by rotating.

No robot, and no position target on the slide. The only coupling is Newton's
mimic constraint. A push and a pull along the stem must not slide it. A torque
about the stem must advance it by the pitch.
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import register_screw_joint_mimic
from isaaclab_hiveboard.tasks.anymal.screw import screw_commands
from isaaclab_hiveboard.tasks.anymal.small_valve.env import AnymalSmallValvePhysicsCfg
from isaaclab_hiveboard.tasks.scenes.small_valve import SMALL_VALVE_SPEC, SmallValveSceneCfg

DT = 1.0 / 200.0
# A firm axial shove, well above the handwheel's weight.
PUSH_N = 30.0
# Off the screw line by more than this [m] is a slide, not a turn.
SLIP_M = 0.0005
# Torque about the stem [N m], just above the joint friction, so the
# handwheel turns about a radian instead of spinning off the thread.
TURN_NM = 0.04
TURN_S = 2.0
# USD joint axis is Z, but the handwheel body's +X is that axis: the joint
# frame is rotated relative to eixo_trans. Confirmed by a free shove.
_STEM_AXIS = (1.0, 0.0, 0.0)


def _valve_cfg() -> object:
    field = SmallValveSceneCfg.__dataclass_fields__["small_valve"]
    base = field.default_factory() if field.default_factory is not None else field.default  # type: ignore[misc]
    return base.replace(prim_path="/World/Env_0/Valve")


def _q(valve: Articulation, index: int) -> float:
    return float(valve.data.joint_pos.torch[0, index].item())


def _wrench(valve: Articulation, body: int, force: tuple[float, float, float], torque: tuple[float, float, float]) -> None:
    """Force [N] and torque [N m] in the handwheel frame."""
    valve.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[list(force)]], device=valve.device),
        torques=torch.tensor([[list(torque)]], device=valve.device),
        body_ids=torch.tensor([body], device=valve.device, dtype=torch.int32),
        is_global=False,
    )


def _run(
    valve: Articulation,
    sim: SimulationContext,
    revolute: int,
    prismatic: int,
    seconds: float,
    *,
    force: tuple[float, float, float] = (0.0, 0.0, 0.0),
    torque: tuple[float, float, float] = (0.0, 0.0, 0.0),
    body: int,
    max_turn: float | None = None,
) -> tuple[float, float, float]:
    """Step with no slide target. Returns end angles and max distance off the screw line."""
    pitch = SMALL_VALVE_SPEC.pitch
    angle_prev = _q(valve, revolute)
    angle0 = angle_prev
    axial = _q(valve, prismatic)
    max_slip = 0.0
    steps = int(seconds / DT)
    for _ in range(steps):
        angle = _q(valve, revolute)
        if max_turn is not None and abs(angle - angle0) >= max_turn:
            break
        delta = math.atan2(math.sin(angle - angle_prev), math.cos(angle - angle_prev))
        axial += delta * pitch / (2.0 * math.pi)
        angle_prev = angle
        _wrench(valve, body, force, torque)
        valve.write_data_to_sim()
        sim.step()
        valve.update(DT)
        slip = abs(_q(valve, prismatic) - axial)
        max_slip = max(max_slip, slip)
    valve.permanent_wrench_composer.reset()
    return _q(valve, revolute), _q(valve, prismatic), max_slip


def main() -> int:
    lines: list[str] = []
    coupling = screw_commands("small_valve", SMALL_VALVE_SPEC).pose_command.screw_coupling
    register_screw_joint_mimic(
        coupling,
        revolute_pos_at_reset=0.0,
        prismatic_pos_at_reset=SMALL_VALVE_SPEC.start,
    )

    sim_utils.create_new_stage()
    sim = SimulationContext(
        SimulationCfg(dt=DT, gravity=(0.0, 0.0, 0.0), physics=AnymalSmallValvePhysicsCfg(), render_interval=1)
    )
    sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))
    valve = Articulation(_valve_cfg())
    sim.reset()
    if not valve.is_initialized:
        raise RuntimeError("small valve articulation did not initialize")

    revolute = valve.find_joints(["RevoluteJoint"])[0][0]
    prismatic = valve.find_joints(["PrismaticJoint"])[0][0]
    body_ids, body_names = valve.find_bodies("eixo_trans")
    if len(body_ids) != 1:
        raise RuntimeError(f"expected one eixo_trans body, found {list(zip(body_ids, body_names))}")
    body = body_ids[0]
    lines.append(f"bodies: {body_names} revolute={revolute} prismatic={prismatic}")

    for _ in range(30):
        valve.write_data_to_sim()
        sim.step()
        valve.update(DT)

    lines.append(f"stem_axis: {_STEM_AXIS}")
    stiffness = float(valve.data.joint_stiffness.torch[0, prismatic].item())
    lines.append(f"prismatic_stiffness: {stiffness:.1f}")

    start_rev, start_pri = _q(valve, revolute), _q(valve, prismatic)
    lines.append(f"settled: revolute={start_rev:+.4f} prismatic={start_pri:+.5f}")

    push_rev, push_pri, push_slip = _run(
        valve, sim, revolute, prismatic, 1.0, force=tuple(PUSH_N * c for c in _STEM_AXIS), body=body
    )
    lines.append(
        f"push +{PUSH_N:.0f}N: revolute={push_rev - start_rev:+.4f} "
        f"prismatic={push_pri - start_pri:+.5f} slip={push_slip:.5f}"
    )
    pull_rev, pull_pri, pull_slip = _run(
        valve, sim, revolute, prismatic, 1.0, force=tuple(-PUSH_N * c for c in _STEM_AXIS), body=body
    )
    lines.append(
        f"pull -{PUSH_N:.0f}N: revolute={pull_rev - push_rev:+.4f} "
        f"prismatic={pull_pri - push_pri:+.5f} slip={pull_slip:.5f}"
    )
    push_ok = push_slip < SLIP_M and pull_slip < SLIP_M
    # A pure axial force must not itself turn or translate the stem.
    push_ok = push_ok and abs(push_pri - start_pri) < SLIP_M and abs(pull_pri - push_pri) < SLIP_M
    push_ok = push_ok and abs(push_rev - start_rev) < 0.05 and abs(pull_rev - push_rev) < 0.05
    lines.append(f"axial_ok: {push_ok}")

    turn_rev, turn_pri, turn_slip = _run(
        valve,
        sim,
        revolute,
        prismatic,
        TURN_S,
        torque=tuple(TURN_NM * c for c in _STEM_AXIS),
        body=body,
        max_turn=1.0,
    )
    d_rev = turn_rev - pull_rev
    d_pri = turn_pri - pull_pri
    expected = d_rev * SMALL_VALVE_SPEC.pitch / (2.0 * math.pi)
    lines.append(
        f"turn {TURN_NM:.2f}Nm for {TURN_S:.1f}s: d_revolute={d_rev:+.4f} "
        f"d_prismatic={d_pri:+.5f} expected={expected:+.5f} slip={turn_slip:.5f}"
    )
    # It has to actually turn, and the stem has to go with it at the pitch.
    turn_ok = abs(d_rev) > 0.3 and abs(d_pri - expected) < SLIP_M and turn_slip < SLIP_M
    lines.append(f"turn_ok: {turn_ok}")

    passed = push_ok and turn_ok
    lines.append("RESULT screw-only" if passed else "RESULT fail")
    text = "\n".join(lines) + "\n"
    args_cli.log.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        args_cli.log.write_text(f"RESULT launch-error\n{traceback.format_exc()}\n", encoding="utf-8")
        print(f"RESULT launch-error\n{traceback.format_exc()}", file=sys.stderr)
        raise
    simulation_app.close()
    raise SystemExit(code)
