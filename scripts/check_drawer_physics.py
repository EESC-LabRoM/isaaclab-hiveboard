# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push the free drawer box and check the side-cut path.

No robot. The assets are the housing and the box ``DrawerSceneCfg`` spawns.
A slow external force, not a joint target, has to slide the box along the
slot, stop on the lip, and leave only by rising there. Up from the closed
pose and from the in-slot open pose stays captured. A modest sideways push
and a modest yaw torque leave both shafts in the walls.
"""

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log", type=Path, required=True)
parser.add_argument("--dt", type=float, help="Override the drawer physics tick in seconds.")
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

import torch
import warp as wp

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import newton
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonManager

import drawer_path as slot
from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN
from isaaclab_hiveboard.tasks.anymal.drawer.physics import DRAWER_SIM_DT, DrawerPhysicsCfg
from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_displacement
from isaaclab_hiveboard.tasks.scenes.drawer import DrawerSceneCfg

DT = DRAWER_SIM_DT if args_cli.dt is None else args_cli.dt
if DT <= 0:
    parser.error("--dt must be positive")
SPEED = 0.02
_ALLOWED_JOINTS = {int(newton.JointType.FREE), int(newton.JointType.FIXED)}


class _World:
    def __init__(self, box: RigidObject, housing: RigidObject):
        self.scene = {"drawer": box, "drawer_housing": housing}


def _cfg(name: str):
    field = DrawerSceneCfg.__dataclass_fields__[name]
    return field.default_factory() if field.default_factory is not None else field.default  # type: ignore[misc]


def _set_mu(mu: float) -> None:
    """Write every shape in this scene. The view slice is empty for a fixed body."""
    model = NewtonManager.get_model()
    binding = model.shape_material_mu
    if binding is None or binding.shape[0] == 0:
        raise RuntimeError("Newton model has no shape friction to set")
    wp.to_torch(binding)[:] = mu
    from newton import ModelFlags

    NewtonManager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


def _pose(world: _World) -> tuple[float, float, float]:
    values = drawer_displacement(world, "drawer", "drawer_housing")[0].detach()
    return tuple(float(v) for v in values.tolist())


def _housing_pos(housing: RigidObject) -> tuple[float, float, float]:
    values = housing.data.root_pos_w.torch[0].detach()
    return tuple(float(v) for v in values.tolist())


def _fmt(trace: list[tuple[float, float, float]]) -> str:
    return " ".join(f"{x:+.4f}/{y:+.4f}/{z:+.4f}" for x, y, z in trace[::4])


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def _default_pose(asset):
    pose = asset.data.default_root_pose
    return pose.warp if hasattr(pose, "warp") else pose


def _default_vel(asset):
    vel = asset.data.default_root_vel
    return vel.warp if hasattr(vel, "warp") else vel


def _reset(box: RigidObject, housing: Articulation, sim: SimulationContext) -> None:
    box.permanent_wrench_composer.reset()
    box.write_root_link_pose_to_sim_index(root_pose=_default_pose(box))
    housing.write_root_link_pose_to_sim_index(root_pose=_default_pose(housing))
    box.write_root_com_velocity_to_sim_index(root_velocity=_default_vel(box))
    housing.write_root_com_velocity_to_sim_index(root_velocity=_default_vel(housing))
    box.write_data_to_sim()
    housing.write_data_to_sim()
    for _ in range(round(0.075 / DT)):
        sim.step()
        box.update(DT)
        housing.update(DT)


def _housing_velocity(box: RigidObject, housing: RigidObject) -> torch.Tensor:
    """Box linear velocity in the housing frame, shape ``(3,)``."""
    velocity_w = box.data.root_lin_vel_w.torch[:1]
    quat = housing.data.root_quat_w.torch[:1]
    return math_utils.quat_apply(math_utils.quat_inv(quat), velocity_w)[0]


def _push(
    box: RigidObject,
    housing: RigidObject,
    sim: SimulationContext,
    seconds: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
    hold_x: float | None = None,
    yaw_torque: float = 0.0,
    notes: list[str] | None = None,
) -> list[tuple[float, float, float]]:
    """Sustained wrench in the box frame along the housing axes.

    The forward push does not ease off. The lip has to stop it.
    """
    world = _World(box, housing)
    trace = []
    steps = int(seconds / DT)
    device = box.device
    for step in range(steps):
        pose = _pose(world)
        vel = _housing_velocity(box, housing)
        x = pose[0]
        cap = 0.08
        if hold_x is None:
            fx = _clamp(6.0 * (vx - float(vel[0])), cap) if vx else 0.0
        else:
            fx = _clamp(8.0 * (hold_x - x) - 0.8 * float(vel[0]), 0.12)
        fy = _clamp(4.0 * (vy - float(vel[1])), 0.02) if vy else 0.0
        fz_cap = 0.012 if hold_x is not None else 0.08
        fz = _clamp(4.0 * (vz - float(vel[2])), fz_cap) if vz else 0.0
        # Body axes match the housing axes at the spawned orientation. The
        # first Newton run confirmed body +X is the measured forward axis.
        force_b = torch.tensor([[[fx, fy, fz]]], device=device)
        omega = box.data.root_ang_vel_w.torch[0]
        torque = (-2.0e-5 * omega).reshape(1, 1, 3).clone()
        torque[0, 0, 2] = torque[0, 0, 2] + yaw_torque
        # The box mass sits on the shaft axis, so a force at the CoM does not pitch it.
        box.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force_b,
            torques=torque,
            is_global=False,
        )
        box.write_data_to_sim()
        sim.step()
        box.update(DT)
        housing.update(DT)
        if notes is not None and step % max(1, round(0.5 / DT)) == 0:
            notes.append(f"t={step * DT:.2f} fx={fx:+.3f} vel={tuple(round(float(v), 4) for v in vel.tolist())} pose={_pose(world)}")
        if step % max(1, round(0.125 / DT)) == 0 or step == steps - 1:
            trace.append(_pose(world))
    box.permanent_wrench_composer.reset()
    return trace


def _drift(start: tuple[float, float, float], housing: RigidObject) -> float:
    now = _housing_pos(housing)
    return sum((a - b) ** 2 for a, b in zip(start, now)) ** 0.5


def main() -> int:
    collision = slot.load_drawer_collision()
    distance = next(iter({shaft.radius for shaft in collision.shafts}))
    lip = slot.forward_end(collision, distance)
    clear_lift = slot.raised_clear(collision, lip, distance)
    lines = [f"lip={lip:.4f} clear_lift={clear_lift:.4f} drawer_open={DRAWER_OPEN:.4f}"]

    sim_utils.create_new_stage()
    sim = SimulationContext(sim_cfg)
    housing_cfg = _cfg("drawer_housing")
    box_cfg = _cfg("drawer")
    # Same orientation for both, so body axes are the housing axes. The scene
    # itself yaws the pair together; that common yaw is not what this check tests.
    housing_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    box_cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    # The scene spawn stays undamped. This check keeps the free box from spinning away.
    box_cfg.spawn.rigid_props.angular_damping = 0.02
    @configclass
    class ContactSceneCfg(InteractiveSceneCfg):
        drawer_housing = housing_cfg
        drawer = box_cfg

    scene = InteractiveScene(ContactSceneCfg(num_envs=1, env_spacing=1.0))
    housing = scene["drawer_housing"]
    box = scene["drawer"]
    sim.reset()
    assert housing.is_initialized and box.is_initialized
    _set_mu(0.05)
    housing.write_data_to_sim()
    box.write_data_to_sim()

    model = NewtonManager.get_model()
    joint_type = model.joint_type
    if joint_type is None or joint_type.shape[0] == 0:
        joint_types = []
        joint_labels = []
    else:
        joint_types = [int(v) for v in joint_type.numpy().tolist()]
        joint_labels = [str(label) for label in model.joint_label]
    lines.append(f"sim_joints: {list(zip(joint_labels, joint_types))}")
    prismatic = [label for label, kind in zip(joint_labels, joint_types) if kind == int(newton.JointType.PRISMATIC)]
    lines.append(f"prismatic: {prismatic}")

    world = _World(box, housing)
    home = _housing_pos(housing)
    _reset(box, housing, sim)
    start = _pose(world)
    lines.append(f"start: {start}")

    drive_notes: list[str] = []
    forward = _push(box, housing, sim, 4.0, vx=SPEED, notes=drive_notes)
    lines.extend(drive_notes)
    held = forward[-1]
    lines.append(f"forward: {_fmt(forward)}")
    # The free pose ends on the lip face. A resting contact sits a few microns
    # inside that face, so the circle test reports a graze. Past the face fails.
    forward_ok = (
        held[0] > 0.012
        and held[0] < lip + 0.0005
        and abs(held[2]) < 0.004
        and slot.y_captured(collision, held)
    )
    lines.append(f"forward_ok: {forward_ok} pose={held}")

    risen = _push(box, housing, sim, 2.5, vx=SPEED, vz=SPEED)
    out = risen[-1]
    lines.append(f"lift_from_lip: {_fmt(risen)}")
    lift_ok = out[2] >= clear_lift - 0.002 and out[0] > held[0] + 0.008 and not slot.hitting(collision, out)
    lines.append(f"lift_ok: {lift_ok} pose={out}")

    _reset(box, housing, sim)
    _push(box, housing, sim, 1.5, hold_x=DRAWER_OPEN)
    open_up = _push(box, housing, sim, 1.5, hold_x=DRAWER_OPEN, vz=SPEED)
    opened = open_up[-1]
    lines.append(f"up_from_open: {_fmt(open_up)}")
    open_captured = (
        abs(opened[0] - DRAWER_OPEN) < 0.004
        and opened[2] < clear_lift * 0.5
        and slot.y_captured(collision, opened)
    )
    lines.append(f"up_from_open_ok: {open_captured} pose={opened}")

    _reset(box, housing, sim)
    only_forward = _push(box, housing, sim, 4.0, vx=SPEED)
    stuck = only_forward[-1]
    lines.append(f"forward_only: {_fmt(only_forward)}")
    forward_only_ok = stuck[0] < lip + 0.0005 and abs(stuck[2]) < 0.004 and slot.y_captured(collision, stuck)
    lines.append(f"forward_only_ok: {forward_only_ok} pose={stuck}")

    _reset(box, housing, sim)
    only_up = _push(box, housing, sim, 1.5, vz=SPEED)
    pinned = only_up[-1]
    lines.append(f"up_only: {_fmt(only_up)}")
    up_only_ok = pinned[2] < clear_lift * 0.5 and pinned[0] < 0.008 and slot.y_captured(collision, pinned)
    lines.append(f"up_only_ok: {up_only_ok} pose={pinned}")

    _reset(box, housing, sim)
    sideways = _push(box, housing, sim, 1.0, vy=0.01)
    slid = sideways[-1]
    lines.append(f"sideways: {_fmt(sideways)}")
    sideways_ok = abs(slid[1]) < 0.002 and slot.y_captured(collision, slid) and abs(slid[0]) < 0.008
    lines.append(f"sideways_ok: {sideways_ok} pose={slid}")

    _reset(box, housing, sim)
    yawed = _push(box, housing, sim, 1.0, yaw_torque=1.0e-5)
    spun = yawed[-1]
    lines.append(f"yaw: {_fmt(yawed)}")
    yaw_ok = slot.y_captured(collision, spun) and abs(spun[0]) < 0.01 and abs(spun[1]) < 0.01
    lines.append(f"yaw_ok: {yaw_ok} pose={spun}")

    housing_ok = _drift(home, housing) < 0.001
    lines.append(f"housing_ok: {housing_ok} drift={_drift(home, housing):.6f}")
    joints_ok = not prismatic and all(kind in _ALLOWED_JOINTS for kind in joint_types)
    lines.append(f"joints_ok: {joints_ok}")

    passed = (
        forward_ok
        and lift_ok
        and open_captured
        and forward_only_ok
        and up_only_ok
        and sideways_ok
        and yaw_ok
        and housing_ok
        and joints_ok
    )
    lines.append("RESULT captured-then-exit" if passed else "RESULT fail")
    text = "\n".join(lines) + "\n"
    args_cli.log.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    sim_cfg = SimulationCfg(
        dt=DT, gravity=(0.0, 0.0, 0.0), physics=DrawerPhysicsCfg().default, render_interval=1
    )
    try:
        with launch_simulation(sim_cfg, args_cli):
            code = main()
    except Exception as exc:
        args_cli.log.write_text(f"RESULT launch-error\n{traceback.format_exc()}\n", encoding="utf-8")
        print(f"RESULT launch-error\n{traceback.format_exc()}", file=sys.stderr)
        raise
    raise SystemExit(code)
