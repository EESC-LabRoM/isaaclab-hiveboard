"""Configuration regression checks for the Franka hidden-button task."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)

import unittest
from types import SimpleNamespace

import gymnasium as gym
import torch

import isaaclab_hiveboard.tasks
from isaaclab_hiveboard.tasks.franka.button.configs.scene import (
    BUTTON_PRESSED_POSITION,
    COVER_CLOSED_POSITION,
    MINIMUM_COVER_SUCCESS_POSITION,
)
from isaaclab_hiveboard.tasks.franka.button.configs.terminations import (
    button_task_success,
)
from isaaclab_hiveboard.tasks.franka.button.env import FrankaButtonEnvCfg


class TestFrankaButton(unittest.TestCase):
    def test_registration_and_configuration(self):
        spec = gym.spec("Isaac-HiveBoard-Franka-Button-v0")
        self.assertTrue(spec.kwargs["env_cfg_entry_point"].endswith(":FrankaButtonEnvCfg"))

        cfg = FrankaButtonEnvCfg()
        cfg.validate()
        self.assertTrue(cfg.scene.honeycomb.spawn.fix_base)
        self.assertTrue(cfg.scene.button.spawn.fix_base)
        self.assertEqual(
            cfg.scene.button.init_state.joint_pos["RevoluteJoint"],
            COVER_CLOSED_POSITION,
        )
        self.assertEqual(
            cfg.scene.button.init_state.joint_pos["PrismaticJoint"], 0.0
        )
        self.assertEqual(
            [
                command.target_frame_name
                for command in cfg.commands.pose_command.commands
                if hasattr(command, "target_frame_name")
            ],
            [
                "cover_clearance",
                "cover_grasp",
                "cover_hinge",
                "post_cover_clearance",
                "button_clearance",
                "button_pressed",
            ],
        )
        self.assertEqual(
            cfg.actions.arm_action.body_offset.rot,
            cfg.commands.pose_command.body_offset.rot,
        )

    def test_success_requires_cover_and_button(self):
        joint_pos = torch.tensor(
            [
                [BUTTON_PRESSED_POSITION, MINIMUM_COVER_SUCCESS_POSITION],
                [BUTTON_PRESSED_POSITION, COVER_CLOSED_POSITION],
                [0.0, MINIMUM_COVER_SUCCESS_POSITION],
            ]
        )
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=joint_pos))
        command = SimpleNamespace(is_done=lambda: torch.ones(3, dtype=torch.bool))
        env = SimpleNamespace(
            scene={"button": asset},
            command_manager=SimpleNamespace(get_term=lambda _: command),
        )
        button_cfg = SimpleNamespace(name="button", joint_ids=[0])
        cover_cfg = SimpleNamespace(name="button", joint_ids=[1])
        success = button_task_success(
            env,
            command_name="pose_command",
            button_cfg=button_cfg,
            cover_cfg=cover_cfg,
            button_tolerance=0.001,
            minimum_cover_angle=MINIMUM_COVER_SUCCESS_POSITION,
        )
        self.assertEqual(success.tolist(), [True, False, False])


if __name__ == "__main__":
    try:
        unittest.main(argv=[__file__])
    finally:
        app_launcher.app.close()
