#!/usr/bin/env python3
# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Edit Spot bench-valve beads in the Newton viewer and re-solve with Isaac Lab IK.

A Newton-side replacement for ``dependencies/hiveboard-bench.github.io/tools/traj_edit.py``.
Select a bead, nudge it, re-solve DLS IK on this robot, play the clip, and save
the same JSON ``play.py`` / ``JointTrajectoryCommand`` already consume.

    uv run python scripts/traj_edit.py physics=newton_mjwarp --visualizer none

Keyboard (Trajectory Editor window focused, not a text field):
    Left / Right    previous / next key
    I/K J/L U/O     translate +/- X/Y/Z (1 cm; Shift = 1 mm)
    Enter           re-solve IK from the selected key
    Ctrl+S          save JSON
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab_hiveboard  # noqa: F401
from isaaclab_hiveboard.assets.spot.bench import ARM_JOINT_NAMES, TRAJECTORY_JSON
from isaaclab_hiveboard.utils.spot_traj import (
    IK_ITERS,
    coincident_indices,
    insert_key_after,
    key_duration_s,
    load_payload,
    n_samples_from_keys,
    nudge_key,
    print_key_report,
    remove_key,
    retarget,
    save_payload,
    set_key_duration_s,
    valve_q_for_sample,
)
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli


DEFAULT_TASK = "Isaac-HiveBoard-Spot-BenchValve-Play-v0"
HELP = """Trajectory editor
  Left/Right  select key     I/K J/L U/O  nudge X/Y/Z
  Shift       1 mm steps     Enter        re-solve IK
  Ctrl+S      save JSON
Beads are the website tcp site. Solve writes q; Play feeds it to the PD clip.
The 3D window is Newton's ViewerGL (not Isaac Lab's visualizer wrapper).
"""


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _register_ui(viewer, callback) -> bool:
    """Attach ``callback`` as a left-panel ``side`` widget (Newton example style)."""
    if not hasattr(viewer, "register_ui_callback"):
        return False
    gui = getattr(viewer, "gui", None)
    callbacks = gui._ui_callbacks.get("side", []) if gui is not None else []
    if callback not in callbacks:
        viewer.register_ui_callback(callback, position="side")
    return True


def _open_newton_viewer(env):
    """Own a Newton ViewerGL so ImGui is the example HUD, not Isaac Lab's wrapper."""
    from isaaclab_newton.physics import NewtonManager
    from newton.viewer import ViewerGL

    viewer = ViewerGL(width=1600, height=900, paused=False)
    gui = getattr(viewer, "gui", None)
    ui = getattr(gui, "ui", None) if gui is not None else None
    if gui is None:
        raise SystemExit("Newton ViewerGL created no GUI object.")
    if ui is not None and not getattr(ui, "is_available", True):
        raise SystemExit(
            "Newton ImGui HUD is disabled (imgui-bundle failed to import). "
            "Check the 'imgui_bundle not found' warning above, then: uv sync"
        )
    model = NewtonManager.get_model()
    if model is None:
        raise SystemExit("NewtonManager has no model; is physics=newton_mjwarp?")
    viewer.set_model(model)
    eye = getattr(env.cfg.viewer, "eye", (1.8, -2.0, 1.2))
    lookat = getattr(env.cfg.viewer, "lookat", (0.7, 0.0, 0.7))
    viewer.camera.pos = viewer.camera._as_vec3(eye)
    viewer.camera.look_at(lookat)
    return viewer


def _key_pressed(imgui, *names: str) -> bool:
    key_enum = imgui.Key
    for name in names:
        key = getattr(key_enum, name, None)
        if key is not None and imgui.is_key_pressed(key):
            return True
    return False


def _key_down(imgui, *names: str) -> bool:
    key_enum = imgui.Key
    for name in names:
        key = getattr(key_enum, name, None)
        if key is not None and imgui.is_key_down(key):
            return True
    return False


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--traj", default=str(TRAJECTORY_JSON), help="Trajectory JSON with keys + q.")
    parser.add_argument("--out", default=None, help="Save path. Default: overwrite --traj.")
    parser.add_argument("--ik-iters", type=int, default=IK_ITERS)
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    if not any(token.startswith(("physics=", "presets=")) for token in hydra_args):
        hydra_args.append("physics=newton_mjwarp")
    # Isaac Lab's wrapper is not used. ``--visualizer none`` must be the
    # AppLauncher form (visualizer=None + explicit + disable_all), not the
    # string ``["none"]``, which SimulationContext treats as an unknown type.
    if args.visualizer not in (None, ["none"], "none"):
        print("[INFO] traj_edit uses Newton ViewerGL directly; ignoring --visualizer", args.visualizer)
    args.visualizer = None
    args.visualizer_explicit = True
    args.visualizer_disable_all = True
    return args, hydra_args


class TrajEditor:
    def __init__(self, env, payload: dict, traj_path: Path, out_path: Path, ik_iters: int):
        self.env = env
        self.payload = payload
        self.keys = payload["keys"]
        self.traj_path = traj_path
        self.out_path = out_path
        self.ik_iters = int(ik_iters)
        self.selected = 0
        self.playing = True
        self.link = True
        self.auto_solve = True
        self.dirty = False
        self.status = "loaded"
        self.report: list[dict] = []
        self._pending: str | None = None
        self._solve_from_selected = True
        self._play_after_solve = False
        command_name = "joint_command" if "joint_command" in env.command_manager.active_terms else "pose_command"
        self.term = env.command_manager.get_term(command_name)
        self.term.set_highlight_key(0)
        self.term.set_frozen(False)
        self._viewer = None

    def current(self) -> dict:
        return self.keys[self.selected]

    def select(self, index: int, preview: bool = True) -> None:
        self.selected = int(index) % len(self.keys)
        self.term.set_highlight_key(self.selected)
        if preview and not self.playing:
            self._pending = "preview"

    def mark_dirty(self) -> None:
        self.dirty = True
        self.status = "unsolved"

    def request_solve(self, from_selected: bool = True) -> None:
        self._solve_from_selected = from_selected
        self._pending = "solve"
        self.status = "solving…"

    def attach_viewer(self, viewer) -> None:
        self._viewer = viewer
        if not _register_ui(viewer, self.render_ui):
            raise SystemExit("Newton viewer has no ImGui callback hook; cannot show the editor.")
        gui = getattr(viewer, "gui", None)
        ui = getattr(gui, "ui", None) if gui is not None else None
        if ui is not None and not getattr(ui, "is_available", True):
            raise SystemExit(
                "Newton ImGui HUD is disabled (imgui-bundle failed to load). "
                "Install it with: uv sync"
            )
        print("[INFO] Trajectory editor: expand 'Example Options' → Trajectory Editor in the left panel.")

    def poll(self) -> None:
        """Run sim-touching work between env.step calls, not inside the UI callback."""
        if self._viewer is not None:
            _register_ui(self._viewer, self.render_ui)
        action = self._pending
        self._pending = None
        if action == "solve":
            self._solve(from_selected=self._solve_from_selected)
        elif action == "save":
            if self.dirty:
                self._solve(from_selected=False)
            self._write()
        elif action == "reload":
            self._reload()
        elif action == "preview":
            self._preview_selected()
        elif action == "play":
            if self.dirty and self.auto_solve:
                self._play_after_solve = True
                self._solve(from_selected=False)
            else:
                self._start_playback()
        elif action == "reset":
            self.env.reset()
            self._start_playback()

    def _start_playback(self) -> None:
        self.term.seek(0)
        self.playing = True
        self.term.set_frozen(False)
        self.status = "playing"

    def _preview_selected(self) -> None:
        sample = int(self.current()["sample"])
        with torch.inference_mode(False):
            self.term.seek(sample)
            self._teleport(sample)

    def pause(self) -> None:
        self.playing = False
        self.term.set_frozen(True)
        if self._pending not in {"solve", "save", "reload", "play", "reset"}:
            self._pending = "preview"
        self.status = "paused"

    def _solve(self, from_selected: bool = True) -> None:
        self.playing = False
        self.term.set_frozen(True)
        start = 0
        if from_selected and self.selected > 0:
            start = int(self.keys[self.selected - 1]["sample"])
        print(f"[INFO] Solving IK from sample {start} ({n_samples_from_keys(self.keys)} samples)…")
        try:
            with torch.inference_mode(False):
                q, report = retarget(self.env, self.payload, self.ik_iters, start_sample=start)
            self.payload["q"] = q
            self.report = report
            print_key_report(report)
            with torch.inference_mode(False):
                self.term.replace_payload(self.payload)
            self.term.set_highlight_key(self.selected)
            self.dirty = False
            self.status = f"solved {len(q)} samples"
            if self._play_after_solve:
                self._play_after_solve = False
                self._start_playback()
            else:
                self._preview_selected()
        except Exception as err:
            self._play_after_solve = False
            self.status = f"solve failed: {err}"
            print(f"[ERROR] {self.status}")

    def _write(self) -> None:
        self.payload["keys"] = self.keys
        self.payload["source"] = f"edited with Isaac Lab Newton traj_edit from {self.traj_path.name}"
        self.payload["keys_source"] = (
            "Newton traj_edit beads + valve-arc finger/approach rotation; Isaac Lab pose IK on website tcp site"
        )
        save_payload(self.out_path, self.payload)
        self.status = f"saved {self.out_path}"
        print(f"[INFO] Wrote {self.out_path}")

    def _reload(self) -> None:
        self.payload = load_payload(self.traj_path)
        self.keys = self.payload["keys"]
        self.selected = min(self.selected, len(self.keys) - 1)
        with torch.inference_mode(False):
            self.term.replace_payload(self.payload)
        self.dirty = False
        self.status = "reloaded"
        self.term.set_highlight_key(self.selected)
        if not self.playing:
            self._preview_selected()

    def _teleport(self, sample: int) -> None:
        robot = self.env.scene["robot"]
        cmd = _as_torch(self.term.command)[0]
        q = _as_torch(robot.data.joint_pos).clone()
        qd = torch.zeros_like(q)
        ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        q[0, ids] = cmd
        robot.write_joint_state_to_sim(q, qd)
        robot.write_data_to_sim()
        robot.update(0.0)
        if "ball_valve" not in self.env.scene.keys():
            return
        valve = self.env.scene["ball_valve"]
        joint_ids, _ = valve.find_joints("RevoluteJoint")
        if not joint_ids:
            return
        vq = _as_torch(valve.data.joint_pos).clone()
        vqd = torch.zeros_like(vq)
        vq[0, joint_ids[0]] = float(valve_q_for_sample(self.keys, sample))
        valve.write_joint_state_to_sim(vq, vqd)
        valve.write_data_to_sim()
        valve.update(0.0)

    def log_beads(self, viewer) -> None:
        import warp as wp

        points, radii, colors = [], [], []
        for index, key in enumerate(self.keys):
            pos = key.get("pos")
            if pos is None:
                continue
            points.append((float(pos[0]), float(pos[1]), float(pos[2])))
            if index == self.selected:
                radii.append(0.024)
                colors.append((0.15, 0.95, 0.75))
            elif key.get("transit") or key.get("home"):
                radii.append(0.015)
                colors.append((0.45, 0.6, 0.85))
            else:
                radii.append(0.018)
                colors.append((0.95, 0.7, 0.15))
        if not points:
            return
        device = getattr(viewer, "device", None)
        viewer.log_points(
            "traj_edit/keys",
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(radii, dtype=wp.float32, device=device),
            wp.array(colors, dtype=wp.vec3, device=device),
        )

    def _nudge(self, delta: np.ndarray) -> None:
        if self.current().get("home"):
            self.status = "home keys are not dragged"
            return
        if self.playing:
            self.pause()
        nudge_key(self.keys, self.selected, delta, link=self.link)
        self.mark_dirty()

    def render_ui(self, imgui) -> None:
        # Drawn inside Newton's left panel (panel/side callbacks). Do not
        # imgui.begin() a nested window — that hides the widgets and can
        # break the rest of the HUD.
        try:
            imgui.set_next_item_open(True, imgui.Cond_.appearing)
            if not imgui.collapsing_header("Trajectory Editor"):
                return
            self._draw_body(imgui)
            self._handle_keys(imgui)
        except Exception:
            import traceback

            traceback.print_exc()
            self.status = "UI error (see terminal)"

    def _draw_body(self, imgui) -> None:
        imgui.text_wrapped(f"{self.traj_path.name}   {self.status}")
        imgui.separator()
        if imgui.button("Play"):
            self._pending = "play"
        imgui.same_line()
        if imgui.button("Restart"):
            self._pending = "reset"
        imgui.same_line()
        if imgui.button("Pause"):
            self.pause()
        imgui.same_line()
        if imgui.button("Solve"):
            self.request_solve()
        imgui.same_line()
        if imgui.button("Save"):
            self._pending = "save"
        if imgui.button("Reload file"):
            self._pending = "reload"
        imgui.same_line()
        changed, self.link = imgui.checkbox("Link stacked beads", self.link)
        changed, self.auto_solve = imgui.checkbox("Auto-solve on edit", self.auto_solve)

        labels = [f"{k.get('id', i)}  {k.get('label', '')}  @{k.get('sample', 0)}" for i, k in enumerate(self.keys)]
        changed, new_idx = imgui.combo("Key", self.selected, labels)
        if changed:
            self.select(int(new_idx))

        if imgui.button("Prev"):
            self.select(self.selected - 1)
        imgui.same_line()
        if imgui.button("Next"):
            self.select(self.selected + 1)
        imgui.same_line()
        if imgui.button("Add after"):
            if self.playing:
                self.pause()
            self.select(insert_key_after(self.keys, self.selected), preview=False)
            self.mark_dirty()
            if self.auto_solve:
                self.request_solve()
        imgui.same_line()
        if imgui.button("Remove"):
            if self.playing:
                self.pause()
            self.select(remove_key(self.keys, self.selected), preview=False)
            self.mark_dirty()
            if self.auto_solve:
                self.request_solve()

        key = self.current()
        imgui.separator()
        imgui.text(f"kind={key.get('kind')}  home={bool(key.get('home'))}  sample={int(key.get('sample', 0))}")
        clip_i = int(self.term.index[0].item())
        imgui.text(f"clip sample {clip_i}/{self.term.num_samples - 1}")

        pos = [float(v) for v in key["pos"]]
        changed_pos = False
        for axis, label in enumerate("xyz"):
            ch, pos[axis] = imgui.input_float(label, pos[axis], 0.001, 0.01, "%.5f")
            changed_pos = changed_pos or ch
            if imgui.is_item_deactivated_after_edit() and self.auto_solve:
                self.request_solve()
        if changed_pos and not key.get("home"):
            if self.playing:
                self.pause()
            new_pos = [round(float(v), 5) for v in pos]
            targets = coincident_indices(self.keys, self.selected) if self.link else [self.selected]
            delta = np.asarray(new_pos, dtype=np.float64) - np.asarray(key["pos"], dtype=np.float64)
            for i in targets:
                if self.keys[i].get("home"):
                    continue
                cur = np.asarray(self.keys[i]["pos"], dtype=np.float64) + delta
                self.keys[i]["pos"] = [round(float(v), 5) for v in cur]
            self.mark_dirty()

        grip = float(key["grip"])
        ch, grip = imgui.slider_float("grip", grip, -1.57, 0.0, "%.3f")
        if ch:
            if self.playing:
                self.pause()
            key["grip"] = round(float(grip), 4)
            self.mark_dirty()
        if imgui.is_item_deactivated_after_edit() and self.auto_solve:
            self.request_solve()

        secs = key_duration_s(self.keys, self.selected)
        ch, secs = imgui.input_float("secs from prev", secs, 0.02, 0.2, "%.3f")
        if ch and self.selected > 0:
            if self.playing:
                self.pause()
            set_key_duration_s(self.keys, self.selected, max(float(secs), 0.02))
            self.mark_dirty()
        if imgui.is_item_deactivated_after_edit() and self.auto_solve:
            self.request_solve()

        if self.report:
            imgui.separator()
            imgui.text("IK residual at beads")
            for row in self.report:
                mark = ">" if str(row.get("id")) == str(key.get("id")) else " "
                imgui.text(f"{mark} {row['id']} {row['label']:<8} {row['err_norm_mm']:.1f} mm  {row['err_rot_deg']:.1f}°")

        imgui.separator()
        imgui.text_wrapped("I/K J/L U/O nudge · Shift 1 mm · Enter solve · Ctrl+S save")

    def _handle_keys(self, imgui) -> None:
        io = imgui.get_io()
        if getattr(io, "want_text_input", False):
            return
        if getattr(io, "want_capture_keyboard", False) and not imgui.is_window_focused():
            return
        step = 0.001 if _key_down(imgui, "left_shift", "right_shift") else 0.01
        if _key_pressed(imgui, "left_arrow"):
            self.select(self.selected - 1)
        elif _key_pressed(imgui, "right_arrow"):
            self.select(self.selected + 1)
        elif _key_pressed(imgui, "i"):
            self._nudge(np.array([step, 0.0, 0.0]))
        elif _key_pressed(imgui, "k"):
            self._nudge(np.array([-step, 0.0, 0.0]))
        elif _key_pressed(imgui, "j"):
            self._nudge(np.array([0.0, step, 0.0]))
        elif _key_pressed(imgui, "l"):
            self._nudge(np.array([0.0, -step, 0.0]))
        elif _key_pressed(imgui, "u"):
            self._nudge(np.array([0.0, 0.0, step]))
        elif _key_pressed(imgui, "o"):
            self._nudge(np.array([0.0, 0.0, -step]))
        elif _key_pressed(imgui, "enter", "keypad_enter"):
            self.request_solve()
        elif _key_down(imgui, "left_ctrl", "right_ctrl") and _key_pressed(imgui, "s"):
            self._pending = "save"


def _route_command(env, command: torch.Tensor) -> torch.Tensor:
    terms = list(env.action_manager.active_terms)
    dims = list(env.action_manager.action_term_dim)
    if terms == ["arm_action"] and dims == [7]:
        return command
    if terms == ["gripper_action", "arm_action"] and dims == [1, 7]:
        return command
    return command


def main() -> int:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 1.0e6
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders = None
    if getattr(env_cfg, "terminations", None) is not None and hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    physics = getattr(env_cfg.sim, "physics", None)
    if physics is not None and hasattr(physics, "use_cuda_graph"):
        physics.use_cuda_graph = False
    if hasattr(env_cfg.sim, "visualizer_cfgs"):
        env_cfg.sim.visualizer_cfgs = None
    # The task follows Cartesian keys; the editor still needs seekable joint
    # playback to preview and retarget its cached q samples.
    if hasattr(env_cfg.commands, "pose_command"):
        from isaaclab_hiveboard.mdp.commands.joint_trajectory_command import JointTrajectoryCommandCfg
        from isaaclab_hiveboard.tasks.spot.bench_valve.configs.actions import SpotBenchJointActionCfg

        env_cfg.commands.pose_command = JointTrajectoryCommandCfg(trajectory_path=args.traj)
        env_cfg.actions = SpotBenchJointActionCfg()
    joint_command = getattr(getattr(env_cfg, "commands", None), "joint_command", None)
    if joint_command is not None:
        joint_command.debug_vis = False
        joint_command.trajectory_path = args.traj

    traj_path = Path(args.traj)
    payload = load_payload(traj_path)
    if "keys" not in payload or "q" not in payload:
        raise SystemExit(f"{traj_path} needs both 'keys' and 'q'.")
    out_path = Path(args.out) if args.out else traj_path

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        base = env.unwrapped
        try:
            obs, _ = env.reset()
            editor = TrajEditor(base, copy.deepcopy(payload), traj_path, out_path, args.ik_iters)
            viewer = _open_newton_viewer(base)
            editor.attach_viewer(viewer)
            viewer.set_reset_callback(lambda: setattr(editor, "_pending", "reset"))
            print(HELP)
            print(f"[INFO] Editing {traj_path}")
            from isaaclab_newton.physics import NewtonManager

            sim_time = 0.0
            step_dt = float(base.cfg.sim.dt) * float(base.cfg.decimation)
            while viewer.is_running():
                editor.poll()
                if viewer.should_step():
                    if (
                        isinstance(obs, dict)
                        and isinstance(obs.get("policy"), dict)
                        and "command" in obs["policy"]
                    ):
                        action = _route_command(base, obs["policy"]["command"])
                    else:
                        action = torch.zeros(env.action_space.shape, device=base.device)
                    with torch.inference_mode():
                        obs, _, terminated, truncated, _ = env.step(action)
                    sim_time += step_dt
                    term = terminated.any().item() if torch.is_tensor(terminated) else bool(terminated)
                    trunc = truncated.any().item() if torch.is_tensor(truncated) else bool(truncated)
                    if (term or trunc) and editor.playing:
                        obs, _ = env.reset()
                        editor.term.set_frozen(False)
                viewer.begin_frame(sim_time)
                viewer.log_state(NewtonManager.get_state())
                editor.log_beads(viewer)
                viewer.end_frame()
            viewer.close()
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
