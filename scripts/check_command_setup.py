# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU regression checks for command authoring and replay settings.

Run with: uv run python scripts/check_command_setup.py
"""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    SequentialPoseCommandCfg,
)
from isaaclab_hiveboard.utils.command_preview import PreviewIK, build_segments, pose_to_viser, viser_to_pose
from isaaclab_hiveboard.utils.command_setup import apply_setup, load_setup, make_setup, save_setup, validate_setup

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import FrameTransformerCfg


def proxy(value):
    return NS(torch=torch.tensor(value, dtype=torch.float32))


def geometry_term():
    robot = NS(data=NS(root_pos_w=proxy([[0, 0, 0]]), root_quat_w=proxy([[0, 0, 0, 1]])))
    sensor = NS(
        data=NS(
            target_frame_names=["goal"],
            target_pos_w=proxy([[[0, 0, 0]]]),
            target_quat_w=proxy([[[0, 0, 0, 1]]]),
        )
    )
    env = NS(scene={"target_frame": sensor}, step_dt=0.05)
    return NS(
        _asset=robot,
        _env=env,
        device="cpu",
        num_envs=1,
        _valve_asset=None,
        _screw_asset=None,
        _screw_revolute_idx=None,
        cfg=NS(debug_vis=False),
    )


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.command = SequentialPoseCommandCfg(
            asset_name="robot",
            body_name="wrist",
            commands=[
                GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", target_offset_pos=(0.1, 0, 0)),
                GripperCommand(open_gripper=False, duration_s=0.7),
                RotateFrameCfg(
                    frame_name="target_frame", target_frame_name="goal", use_valve_angle=False, angle_deg=40
                ),
            ],
        )

    def test_round_trip_preserves_command_types_and_planner_settings(self):
        self.command.commands.append(
            CuroboPlannedGoToFrameCfg(
                frame_name="target_frame",
                target_frame_name="goal",
                robot_joint_names=["a", "b"],
                num_ik_seeds=7,
            )
        )
        data = make_setup("test-task", self.command)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setup.json"
            save_setup(path, data)
            loaded = load_setup(path)
            commands, _ = validate_setup(loaded, task="test-task")
        self.assertIsInstance(commands[-1], CuroboPlannedGoToFrameCfg)
        self.assertEqual(commands[-1].num_ik_seeds, 7)
        self.assertEqual(commands[1].duration_s, 0.7)
        self.assertFalse(commands[2].use_valve_angle)
        self.assertNotIn("class_type", loaded["commands"][0]["parameters"])

    def test_apply_synchronizes_offsets_and_rejects_bad_reference_without_mutation(self):
        action = DifferentialInverseKinematicsActionCfg(asset_name="robot", body_name="wrist")
        cfg = NS(
            commands=NS(pose_command=self.command),
            actions=NS(arm_action=action),
            scene=NS(
                target_frame=FrameTransformerCfg(target_frames=[FrameTransformerCfg.FrameCfg(name="goal")]),
                ee_frame=FrameTransformerCfg(target_frames=[FrameTransformerCfg.FrameCfg(name="ee_tcp")]),
            ),
        )
        data = make_setup("test-task", self.command)
        data["body_offset"] = {"pos": [0.2, 0.1, 0.3], "rot": [0, 0, 2, 2]}
        apply_setup(cfg, data, task="test-task")
        self.assertEqual(action.body_offset.pos, (0.2, 0.1, 0.3))
        self.assertEqual(action.body_offset.rot, cfg.scene.ee_frame.target_frames[0].offset.rot)
        self.assertEqual(action.body_offset.rot, cfg.commands.pose_command.body_offset.rot)
        data["commands"][0]["parameters"]["target_frame_name"] = "missing"
        before = make_setup("test-task", cfg.commands.pose_command)
        with self.assertRaisesRegex(ValueError, "Unknown reference"):
            apply_setup(cfg, data, task="test-task")
        self.assertEqual(before, make_setup("test-task", cfg.commands.pose_command))

    def test_invalid_file_cannot_overwrite_valid_setup(self):
        good = make_setup("test-task", self.command)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "setup.json"
            save_setup(path, good)
            original = path.read_bytes()
            bad = copy.deepcopy(good)
            bad["body_offset"]["rot"] = [0, 0, 0, 0]
            with self.assertRaises(ValueError):
                save_setup(path, bad)
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "belongs to"):
            validate_setup(good, task="different-task")
        for field, value in (("duration_s", -1), ("open_gripper", "false")):
            bad = copy.deepcopy(good)
            bad["commands"][1]["parameters"][field] = value
            with self.assertRaises(ValueError):
                validate_setup(bad)
        bad = copy.deepcopy(good)
        bad["commands"][0]["parameters"]["class_type"] = "arbitrary.module:callable"
        with self.assertRaises(ValueError):
            validate_setup(bad)


class GeometryTests(unittest.TestCase):
    def test_unsampled_reference_is_rejected_instead_of_targeting_world_origin(self):
        term = geometry_term()
        term._env.scene["target_frame"].data.target_quat_w = proxy([[[0, 0, 0, 0]]])
        cfg = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=False)
        initial = (torch.zeros(1, 3), torch.tensor([[0., 0, 0, 1]]))
        with self.assertRaisesRegex(ValueError, "valid quaternion"):
            build_segments(term, [cfg], initial)

    def test_goal_offset_rotates_with_reference_and_robot_base(self):
        term = geometry_term()
        term._asset.data.root_pos_w = proxy([[1, 0, 0]])
        term._asset.data.root_quat_w = proxy([[0, 0, math.sqrt(0.5), math.sqrt(0.5)]])
        sensor = term._env.scene["target_frame"]
        sensor.data.target_pos_w = proxy([[[1, 2, 0]]])
        sensor.data.target_quat_w = proxy([[[0, 0, math.sqrt(0.5), math.sqrt(0.5)]]])
        cfg = GoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            canonicalize_upward=False,
            target_offset_pos=(0.5, 0, 0),
            target_offset_rot=(0, 0, 1, 0),
        )
        initial = (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1]]))
        segment = build_segments(term, [cfg], initial)[0]
        torch.testing.assert_close(segment.end[0], torch.tensor([[2.5, 0.0, 0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(segment.end[1].abs(), torch.tensor([[0.0, 0, 1, 0]]), atol=1e-6, rtol=0)

    def test_rotate_and_screw_match_runtime_geometry_at_signed_half_arc(self):
        term = geometry_term()
        initial = (torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        for cls in (RotateFrameCfg, ScrewFrameCfg):
            cfg = cls(
                frame_name="target_frame",
                target_frame_name="goal",
                axis=(0, 0, 1),
                angle_deg=-180,
                use_valve_angle=False,
            )
            if isinstance(cfg, ScrewFrameCfg):
                cfg.axial_distance = 0.4
            segment = build_segments(term, [cfg], initial)[0]
            pos, quat = segment.sample(0.5)
            z = 0.2 if isinstance(cfg, ScrewFrameCfg) else 0.0
            torch.testing.assert_close(pos, torch.tensor([[0.0, -1, z]]), atol=1e-6, rtol=0)
            expected = torch.tensor([[0.0, 0, -math.sqrt(0.5), math.sqrt(0.5)]])
            torch.testing.assert_close(quat, expected, atol=1e-6, rtol=0)

    def test_valve_angle_switch_and_gripper_hold(self):
        term = geometry_term()
        term._valve_asset = object()
        term.valve_rotate_angle_rad = torch.tensor([0.4])
        term.recompute_valve_rotate_angle = lambda ids: None
        initial = (torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        cfg = RotateFrameCfg(frame_name="target_frame", target_frame_name="goal", axis=(0, 0, 1), angle_deg=90)
        segments = build_segments(term, [cfg, GripperCommand()], initial)
        self.assertAlmostEqual(float(segments[0].handler.angle_rad_tensor[0]), 0.4)
        torch.testing.assert_close(segments[1].sample(0.5)[0], segments[0].end[0])
        cfg.use_valve_angle = False
        segment = build_segments(term, [cfg], initial)[0]
        self.assertAlmostEqual(float(segment.handler.angle_rad_tensor[0]), math.pi / 2, places=6)

    def test_viser_quaternion_order_round_trip(self):
        pose = (torch.tensor([[1.0, 2, 3]]), torch.tensor([[0.5, -0.5, 0.5, -0.5]]))
        converted = pose_to_viser(pose)
        result = viser_to_pose(converted["position"], converted["wxyz"], device="cpu")
        for original, restored in zip(pose, result):
            torch.testing.assert_close(original, restored)

    def test_ik_tracks_offset_on_rotating_flange_and_reports_final_residual(self):
        class RotatingWrist:
            is_fixed_base, num_base_dofs = True, 0

            def __init__(self):
                jacobian = torch.zeros(1, 1, 6, 1)
                jacobian[0, 0, 5, 0] = 1
                self.data = NS(
                    joint_pos=proxy([[0]]),
                    root_pos_w=proxy([[0, 0, 0]]),
                    root_quat_w=proxy([[0, 0, 0, 1]]),
                    joint_pos_limits=proxy([[[-math.pi, math.pi]]]),
                    body_link_jacobian_w=NS(torch=jacobian),
                )
                self.update(0)

            def find_joints(self, names, preserve_order=True):
                return [0], ["wrist"]

            def write_joint_state_to_sim(self, q, qd):
                self.data.joint_pos.torch = q.clone()

            def update(self, dt):
                angle = self.data.joint_pos.torch[0, 0]
                self.data.body_pos_w = proxy([[[0, 0, 0], [0, 0, 0]]])
                self.data.body_quat_w = proxy([[[0, 0, 0, 1], [0, 0, math.sin(angle / 2), math.cos(angle / 2)]]])

        robot = RotatingWrist()
        term = NS(
            _body_idx=1, cfg=NS(asset_name="robot", body_offset=SequentialPoseCommandCfg.OffsetCfg(pos=(0.2, 0, 0)))
        )
        env = NS(
            scene={"robot": robot},
            action_manager=NS(
                active_terms=["arm_action"],
                get_term=lambda name: NS(cfg=NS(joint_names=["wrist"])),
            ),
        )
        ik = PreviewIK(env, term)
        a = math.pi / 3
        target = (
            torch.tensor([[0.2 * math.cos(a), 0.2 * math.sin(a), 0]]),
            torch.tensor([[0.0, 0, math.sin(a / 2), math.cos(a / 2)]]),
        )
        pos_error, rot_error = ik.solve(target, False)
        self.assertLess(pos_error, 0.001)
        self.assertLess(rot_error, 0.1)
        self.assertAlmostEqual(float(robot.data.joint_pos.torch[0, 0]), a, places=3)


if __name__ == "__main__":
    unittest.main()
