"""Finite-difference regression for the lever-valve TCP Jacobian.

Run with the project's Isaac Lab Python:
    python tests/test_franka_lever_valve_ik.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)

import unittest
from types import SimpleNamespace

import torch

import isaaclab.utils.math as math_utils
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import _GripperHandler
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import FrankaIKAbsActionCfg
from isaaclab_hiveboard.tasks.franka.lever_valve.configs.actions import FrankaLeverValveIKAction
from isaaclab_hiveboard.tasks.franka.lever_valve.env import FrankaLeverValveEnvCfg
from isaaclab_hiveboard.tasks.scenes.lever_valve import LeverValveSceneCfg


class TestFrankaLeverValveJacobian(unittest.TestCase):
    def test_offset_jacobian_matches_tcp_motion(self):
        """Differentiate TCP poses independently of the offset-Jacobian formula."""
        rng = torch.Generator().manual_seed(42)
        n, joints = 16, 7

        def rand(*shape):
            return torch.randn(*shape, generator=rng, dtype=torch.float64)

        def random_quat():
            quat = rand(n, 4)
            return quat / quat.norm(dim=-1, keepdim=True)

        root_quat, body_quat, offset_quat = [random_quat() for _ in range(3)]
        body_pos, offset_pos = rand(n, 3), 0.1 * rand(n, 3)
        # Include a rotated tool with zero displacement.
        offset_pos[0] = 0.0
        jacobian_b = rand(n, 6, joints)
        rotation_w = math_utils.matrix_from_quat(root_quat)
        jacobian_w = torch.cat(
            [rotation_w @ jacobian_b[:, :3], rotation_w @ jacobian_b[:, 3:]], dim=1
        )
        action = object.__new__(FrankaLeverValveIKAction)
        action._debug_vis_handle = None
        action.cfg = SimpleNamespace(body_offset=SimpleNamespace())
        action._body_idx = 0
        action._jacobi_body_idx = 0
        action._jacobi_joint_ids = list(range(joints))
        action._offset_pos, action._offset_rot = offset_pos, offset_quat
        action._asset = SimpleNamespace(
            data=SimpleNamespace(
                root_quat_w=root_quat,
                body_quat_w=math_utils.quat_mul(root_quat, body_quat)[:, None],
            ),
            root_physx_view=SimpleNamespace(get_jacobians=lambda: jacobian_w[:, None].clone()),
        )
        analytic = action._compute_frame_jacobian()
        numeric = torch.empty_like(analytic)
        eps = 1.0e-5
        for joint in range(joints):
            poses = []
            for sign in (-1, 1):
                perturbed_quat = math_utils.quat_box_plus(
                    body_quat, sign * eps * jacobian_b[:, 3:, joint], eps=1.0e-12
                )
                poses.append(math_utils.combine_frame_transforms(
                    body_pos + sign * eps * jacobian_b[:, :3, joint],
                    perturbed_quat, offset_pos, offset_quat,
                ))
            numeric[:, :3, joint] = (poses[1][0] - poses[0][0]) / (2 * eps)
            numeric[:, 3:, joint] = math_utils.quat_box_minus(poses[1][1], poses[0][1]) / (2 * eps)
        torch.testing.assert_close(analytic, numeric, atol=1.0e-7, rtol=1.0e-6)
        # Reading the Jacobian repeatedly must not accumulate the TCP offset.
        torch.testing.assert_close(action._compute_frame_jacobian(), analytic)
        action.cfg.body_offset = None
        torch.testing.assert_close(action._compute_frame_jacobian(), jacobian_b)

    def test_correction_is_scoped_to_lever_valve(self):
        cfg = FrankaLeverValveEnvCfg()
        self.assertIs(cfg.actions.arm_action.class_type, FrankaLeverValveIKAction)
        self.assertIsNot(FrankaIKAbsActionCfg().arm_action.class_type, FrankaLeverValveIKAction)
        self.assertEqual(cfg.actions.arm_action.body_offset.rot, cfg.commands.pose_command.body_offset.rot)
        self.assertEqual(cfg.actions.arm_action.body_offset.rot, cfg.scene.ee_frame.target_frames[0].offset.rot)

    def test_retreat_is_fixed_to_housing_and_follows_release(self):
        cfg = FrankaLeverValveEnvCfg()
        release, retreat = cfg.commands.pose_command.commands[-2:]
        self.assertTrue(release.open_gripper)
        self.assertGreaterEqual(release.duration_s, 0.3)
        frame = next(f for f in cfg.scene.target_frame.target_frames if f.name == retreat.target_frame_name)
        self.assertTrue(frame.prim_path.endswith("/valvula_esfera"))
        self.assertFalse(retreat.hold_current_orientation)
        self.assertNotIn("retreat", [f.name for f in LeverValveSceneCfg().target_frame.target_frames])

    def test_gripper_debug_target_is_the_held_tcp_pose(self):
        handler = object.__new__(_GripperHandler)
        command = torch.tensor([
            [1.0, 0.52, 0.0, 0.34, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.43, 0.0, 0.40, 0.0, 0.0, 0.0, 1.0],
        ])
        handler._command_term = SimpleNamespace(command=command)
        handler._asset = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3), root_quat_w=torch.zeros(2, 4),
        ))
        pos, quat = handler.get_target_in_base_frame(torch.tensor([1]))
        torch.testing.assert_close(pos, command[1:2, 1:4])
        torch.testing.assert_close(quat, command[1:2, 4:8])


if __name__ == "__main__":
    try:
        unittest.main(argv=[__file__])
    finally:
        app_launcher.app.close()
