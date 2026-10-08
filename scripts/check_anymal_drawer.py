# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the ANYmal expert and verify it grasps, lifts, and removes the drawer.

    uv run python scripts/check_anymal_drawer.py --visualizer none
    uv run python scripts/check_anymal_drawer.py --visualizer none \
        --task Isaac-HiveBoard-Anymal-Drawer-v0 --setup configs/Isaac-HiveBoard-Anymal-Drawer-v0.json
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import torch
from pxr import Gf, Usd, UsdGeom

import isaaclab_hiveboard  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_newton.physics import NewtonManager
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN
from isaaclab_hiveboard.tasks.anymal.drawer.slide import drawer_clear_of_guides, drawer_displacement
from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup


def _fingertip_meshes(robot):
    """Read actual fingertip vertices in their moving finger body's frame."""
    stage = Usd.Stage.Open(robot.cfg.spawn.usd_path)
    cache = UsdGeom.XformCache()
    bodies = {prim.GetName(): prim for prim in stage.Traverse() if prim.GetName().endswith("inner_finger")}
    meshes = []
    for prim in stage.Traverse():
        if prim.GetName() != "Fingertip" or not prim.IsA(UsdGeom.Mesh):
            continue
        name = "left_inner_finger" if "/left_inner_finger/" in str(prim.GetPath()) else "right_inner_finger"
        transform = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(bodies[name]).GetInverse()
        vertices = torch.tensor(
            [tuple(transform.Transform(Gf.Vec3d(*point))) for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()],
            device=robot.device,
        )
        meshes.append((robot.find_bodies(name)[0][0], vertices))
    if len(meshes) != 2:
        raise ValueError("Expected both Robotiq fingertip meshes.")
    return meshes


def _tip_insertion(base, meshes) -> float:
    """Maximum fingertip depth behind the drawer's inner front plane [m]."""
    robot = base.scene["robot"]
    drawer = base.scene["drawer"]
    min_x = float("inf")
    for body, points in meshes:
        quat = robot.data.body_quat_w.torch[0, body].expand(len(points), -1)
        world = math_utils.quat_apply(quat, points) + robot.data.body_pos_w.torch[0, body]
        local = math_utils.quat_apply_inverse(
            drawer.data.root_quat_w.torch[0].expand(len(points), -1), world - drawer.data.root_pos_w.torch[0]
        )
        min_x = min(min_x, float(local[:, 0].min()))
    # The drawer front meets the housing at local x=0.038 (metres).
    return max(0.0, 0.038 - min_x)


def _both_jaws_on_handle() -> bool:
    contacts = NewtonManager.get_contacts()
    model = NewtonManager.get_model()
    count = int(contacts.rigid_contact_count.numpy()[0])
    jaws = set()
    for a, b in zip(contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]):
        labels = (model.shape_label[a], model.shape_label[b])
        if any("/Drawer/Geometry/drawer/face" in label for label in labels):
            for name in ("left_inner_finger", "right_inner_finger"):
                if any(f"/{name}/" in label for label in labels):
                    jaws.add(name)
    return len(jaws) == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("Isaac-HiveBoard-Anymal-Drawer-v0", "Isaac-HiveBoard-Anymal-Drawer-Play-v0"),
        default="Isaac-HiveBoard-Anymal-Drawer-Play-v0",
    )
    parser.add_argument("--setup", help="Command-editor JSON; omit to check the environment defaults.")
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in overrides):
        overrides.append("physics=newton_mjwarp")
    sys.argv = [sys.argv[0], *overrides]
    cfg, _ = resolve_task_config(args.task, "")
    if args.setup:
        apply_setup(cfg, load_setup(args.setup), task=args.task)
    cfg.scene.num_envs = 1
    cfg.commands.pose_command.stall_timeout_s = 4.0
    cfg.recorders = None
    cfg.seed = args.seed
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        try:
            base = env.unwrapped
            robot = base.scene["robot"]
            term = base.command_manager.get_term("pose_command")
            finger = robot.find_joints("finger_joint")[0][0]
            opening = cfg.actions.gripper_action.open_command_expr["finger_joint"]
            closing = cfg.actions.gripper_action.close_command_expr["finger_joint"]
            meshes = _fingertip_meshes(robot)
            max_insertion = 0.0
            max_travel = 0.0
            max_vertical = 0.0
            grasped = False
            removed_while_grasped = False
            grasp_reference = None
            max_grasp_slip = 0.0
            previous_segment = None
            fallback = False
            obs, _ = env.reset(seed=args.seed)
            for step in range(base.max_episode_length):
                with torch.inference_mode():
                    seg = int(term._current_command_idx[0])
                    grip = float(robot.data.joint_pos.torch[0, finger])
                    # Closing must stop on the tab between the open and closed
                    # motor targets, rather than reaching the empty-hand stop.
                    displacement = drawer_displacement(base, "drawer", "drawer_housing")[0]
                    holding = opening + 0.005 < grip < closing - 0.005
                    if 3 <= seg <= 7:
                        max_vertical = max(max_vertical, float(displacement[2]))
                        grasped |= opening + 0.005 < grip < closing - 0.005
                        max_insertion = max(max_insertion, _tip_insertion(base, meshes))
                        removed_while_grasped |= (
                            holding and bool(drawer_clear_of_guides(base, "drawer", "drawer_housing")[0])
                            and _both_jaws_on_handle()
                        )
                        drawer = base.scene["drawer"]
                        tcp_w = base.scene["ee_frame"].data.target_pos_w.torch[0, 0]
                        tcp_local = math_utils.quat_apply_inverse(
                            drawer.data.root_quat_w.torch[0], tcp_w - drawer.data.root_pos_w.torch[0]
                        )
                        if grasp_reference is None:
                            grasp_reference = tcp_local.clone()
                        max_grasp_slip = max(max_grasp_slip, float((tcp_local - grasp_reference).norm()))
                    if seg != previous_segment or (seg >= 3 and step % 10 == 0):
                        quat = base.scene["drawer"].data.root_quat_w.torch[0]
                        housing_quat = base.scene["drawer_housing"].data.root_quat_w.torch[0]
                        local_quat = math_utils.quat_mul(math_utils.quat_inv(housing_quat), quat)
                        rpy = [float(torch.rad2deg(v)[0]) for v in math_utils.euler_xyz_from_quat(local_quat[None])]
                        print(f"[DRAWER TRACE] t={step * base.step_dt:.2f} segment={seg} "
                              f"pose={displacement.tolist()} rpy={rpy} grip={grip:.4f}", flush=True)
                        previous_segment = seg
                        if seg == 4 and step % 30 == 0:
                            contacts = NewtonManager.get_contacts()
                            model = NewtonManager.get_model()
                            count = int(contacts.rigid_contact_count.numpy()[0])
                            pairs = {(model.shape_label[a], model.shape_label[b]) for a, b in zip(
                                contacts.rigid_contact_shape0.numpy()[:count],
                                contacts.rigid_contact_shape1.numpy()[:count],
                            ) if "/Drawer/" in model.shape_label[a] or "/Drawer/" in model.shape_label[b]}
                            print(f"[DRAWER CONTACTS] {sorted(pairs)}", flush=True)
                    max_travel = max(max_travel, float(displacement[0]))
                    fallback |= bool(term.expert_fallback()[0])
                    obs, _, terminated, truncated, info = env.step(obs["policy"]["command"])
                    if bool(terminated[0] or truncated[0]):
                        success = bool(info["log"]["Episode_Termination/success"])
                        passed = (
                            success and grasped and max_travel >= DRAWER_OPEN and not fallback
                            and max_insertion <= 0.002 and max_vertical >= 0.01
                            and removed_while_grasped and max_grasp_slip <= 0.01
                        )
                        print(
                            f"[DRAWER] {'PASS' if passed else 'FAIL'} task={args.task} seed={args.seed} "
                            f"time={(step + 1) * base.step_dt:.2f}s travel={max_travel * 1000:.1f}mm "
                            f"grasped={grasped} success={success} fallback={fallback} "
                            f"tip_insertion={max_insertion * 1000:.1f}mm lift={max_vertical * 1000:.1f}mm "
                            f"removed_while_grasped={removed_while_grasped} slip={max_grasp_slip * 1000:.1f}mm",
                            flush=True,
                        )
                        return 0 if passed else 1
            print("[DRAWER] FAIL: trajectory did not finish within the episode.", flush=True)
            return 1
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
