"""Run with the Isaac Lab Python; add --rollout for a physical smoke test."""

import sys

from isaaclab.app import AppLauncher

rollout = "--rollout" in sys.argv
app_launcher = AppLauncher(headless=True)

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import torch

import isaaclab_hiveboard.tasks
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.progress import (
    GateValvePoseCommand,
    accumulate_rotation,
    full_turn_success,
)
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import SequentialPoseCommand
from isaaclab_hiveboard.tasks.franka.gate_valve_small.env import FrankaGateValveSmallEnvCfg
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import FRAME_MARKER_SMALL_CFG


class TestGateValveSmall(unittest.TestCase):
    def test_registration_and_configuration(self):
        spec = gym.spec("Isaac-HiveBoard-Franka-GateValveSmall-v0")
        self.assertTrue(spec.kwargs["env_cfg_entry_point"].endswith(":FrankaGateValveSmallEnvCfg"))
        cfg = FrankaGateValveSmallEnvCfg()
        cfg.validate()
        self.assertEqual(cfg.scene.small_valve.init_state.joint_pos["RevoluteJoint"], 0.0)
        self.assertTrue(cfg.scene.small_valve.spawn.fix_base)
        self.assertTrue(cfg.scene.honeycomb.spawn.fix_base)
        self.assertEqual(cfg.actions.arm_action.body_offset.rot, cfg.commands.pose_command.body_offset.rot)
        self.assertEqual(cfg.scene.target_frame.visualizer_cfg.markers["frame"].scale,
                         cfg.scene.ee_frame.visualizer_cfg.markers["frame"].scale)
        self.assertEqual(cfg.scene.target_frame.visualizer_cfg.markers["frame"].scale,
                         FRAME_MARKER_SMALL_CFG.markers["frame"].scale)
        commands = cfg.commands.pose_command.commands
        self.assertEqual(len(commands), 29)
        for start in range(1, 29, 7):
            engage, grasp, rotate, settle, release, lift, unwind = commands[start:start + 7]
            self.assertFalse(grasp.open_gripper)
            self.assertEqual(rotate.angle_deg, 92.0)
            self.assertEqual(rotate.axis, (0.0, -1.0, 0.0))
            self.assertTrue(release.open_gripper)
            self.assertTrue(lift.hold_current_orientation)
            self.assertFalse(unwind.hold_current_orientation)

    def test_success_distinguishes_zero_full_turn_and_backtracking(self):
        angles = torch.tensor([0.0, math.tau, -math.tau, math.pi, 2 * math.tau])
        command = SimpleNamespace(stem_rotation=angles, is_done=lambda: torch.ones(5, dtype=torch.bool))
        env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        kwargs = dict(command_name="pose_command", tolerance=math.radians(5))
        self.assertEqual(full_turn_success(env, **kwargs).tolist(),
                         [False, True, False, False, False])
        command.is_done = lambda: torch.zeros(5, dtype=torch.bool)
        self.assertFalse(full_turn_success(env, **kwargs).any())

    def test_wrapping_and_reverse_motion(self):
        previous = torch.zeros(2)
        accumulated = torch.zeros(2)
        for angle in torch.linspace(0, math.tau, 101):
            current = torch.remainder(torch.tensor([angle, -angle]) + math.pi, math.tau) - math.pi
            accumulated = accumulate_rotation(previous, current, accumulated)
            previous = current
        torch.testing.assert_close(accumulated, torch.tensor([math.tau, -math.tau]))
        backwards = torch.tensor([-0.2, 0.2])
        torch.testing.assert_close(accumulate_rotation(previous, backwards, accumulated),
                                   torch.tensor([math.tau - 0.2, -math.tau + 0.2]))

    def test_partial_reset_preserves_other_environments(self):
        command = object.__new__(GateValvePoseCommand)
        command._debug_vis_handle = None
        command._previous_angle = torch.tensor([1.0, 2.0])
        command.stem_rotation = torch.tensor([3.0, 4.0])
        with patch.object(SequentialPoseCommand, "_resample_command"):
            command._resample_command(torch.tensor([1]))
        torch.testing.assert_close(command._previous_angle, torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(command.stem_rotation, torch.tensor([3.0, 0.0]))


def run_rollout():
    cfg = FrankaGateValveSmallEnvCfg()
    env = gym.make("Isaac-HiveBoard-Franka-GateValveSmall-v0", cfg=cfg)
    try:
        env.reset()
        base = env.unwrapped
        valve = base.scene["small_valve"]
        joint_ids, _ = valve.find_joints("RevoluteJoint")
        with torch.inference_mode():
            for step in range(int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))):
                action = base.command_manager.get_command("pose_command")
                _, _, terminated, truncated, _ = env.step(action)
                if step % 100 == 0:
                    term = base.command_manager.get_term("pose_command")
                    print("ROLLOUT", step, valve.data.joint_pos[:, joint_ids].tolist(),
                          "stage", term._current_command_idx.tolist(),
                          "vel", valve.data.joint_vel[:, joint_ids].tolist(),
                          "travel", term.stem_rotation.tolist(),
                          "tcp", base.scene["ee_frame"].data.target_pos_w.tolist(), flush=True)
                if terminated.any() or truncated.any():
                    print("RESULT", terminated.tolist(), truncated.tolist(), flush=True)
                    if not terminated.all():
                        raise AssertionError("Valve did not complete a physical full turn")
                    break
            else:
                raise AssertionError("Episode did not complete")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestGateValveSmall))
        if not result.wasSuccessful():
            raise AssertionError("Gate valve regression checks failed")
        if rollout:
            run_rollout()
        print("GATE_VALVE_CHECKS_PASSED", flush=True)
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app_launcher.app.close()
