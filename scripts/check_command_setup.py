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
from unittest.mock import patch

import torch
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GoToFrameCfg,
    GripperCommand,
    RotateFrameCfg,
    ScrewFrameCfg,
    SequentialPoseCommandCfg,
    _CuroboPlannedRotateFrameHandler,
    _GoToFrameHandler,
    _GripperHandler,
)
from isaaclab_hiveboard.utils.command_path import active_command_path
from isaaclab_hiveboard.utils.command_preview import (
    PreviewIK,
    Segment,
    apply_authored_command_field,
    build_segments,
    concat_segment_tcp,
    effective_arc_caption,
    format_editor_plan_failure,
    format_infeasible_rotation_message,
    invalidate_preview_plans_from,
    _plan_curobo_per_segment,
    pose_to_viser,
    viser_to_pose,
)
from isaaclab_hiveboard.utils.command_setup import (
    apply_setup,
    as_curobo_command,
    as_direct_command,
    is_curobo_command,
    load_setup,
    make_setup,
    planner_settings,
    save_setup,
    validate_setup,
)
from isaaclab_hiveboard.utils.frame_sensors import refresh_frame_sensors

import isaaclab.utils.math as math_utils
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

    def test_curobo_toggle_preserves_cartesian_fields_and_planner(self):
        planned = CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            target_offset_pos=(0.1, 0.0, 0.0),
            robot_joint_names=["arm_sh0", "arm_sh1"],
            robot_curobo_yaml="spot.yaml",
            num_ik_seeds=8,
        )
        servo = as_direct_command(planned)
        self.assertIsInstance(servo, GoToFrameCfg)
        self.assertFalse(is_curobo_command(servo))
        self.assertEqual(servo.target_offset_pos, (0.1, 0.0, 0.0))
        restored = as_curobo_command(servo, planner_settings([planned]))
        self.assertIsInstance(restored, CuroboPlannedGoToFrameCfg)
        self.assertEqual(restored.target_offset_pos, (0.1, 0.0, 0.0))
        self.assertEqual(restored.robot_joint_names, ["arm_sh0", "arm_sh1"])
        self.assertEqual(restored.num_ik_seeds, 8)
        rotate = RotateFrameCfg(
            frame_name="target_frame", target_frame_name="goal", use_valve_angle=False, angle_deg=40
        )
        planned_rotate = as_curobo_command(rotate, planner_settings([planned]))
        self.assertTrue(is_curobo_command(planned_rotate))
        self.assertEqual(planned_rotate.angle_deg, 40)
        self.assertEqual(planned_rotate.robot_joint_names, ["arm_sh0", "arm_sh1"])
        with self.assertRaisesRegex(ValueError, "Cannot use a cuRobo plan"):
            as_curobo_command(GripperCommand(), planner_settings([planned]))

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
    def test_upright_goal_stays_continuous_across_horizontal_and_can_complete(self):
        term = geometry_term()
        sensor = term._env.scene["target_frame"]
        axis = torch.tensor([[1.0, 0, 0]])
        for sign in (-1, 1):

            def rotation(degrees):
                return math_utils.quat_from_angle_axis(torch.tensor([math.radians(sign * degrees)]), axis)

            sensor.data.target_quat_w.torch[:] = rotation(89.9)[:, None]
            initial = (torch.zeros(1, 3), rotation(89.9))
            cfg = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=True)
            handler = build_segments(term, [cfg], initial)[0].handler
            ids = torch.tensor([0])
            for degrees in (90.1, 89.95, 90.05, 89.9):
                expected = rotation(degrees)
                sensor.data.target_quat_w.torch[:] = expected[:, None]
                _, actual = handler.get_target_in_base_frame(ids)
                torch.testing.assert_close(math_utils.quat_error_magnitude(actual, expected), torch.zeros(1))
                # The measured TCP remains within 0.2 degrees; completion must
                # not suddenly require turning to the opposite grasp.
                self.assertTrue(handler.is_done(ids).item())

    def test_upright_choice_is_per_environment_and_reselected_on_reset(self):
        term = geometry_term()
        term.num_envs = 2
        term._asset.data.root_pos_w = proxy([[0, 0, 0], [0, 0, 0]])
        term._asset.data.root_quat_w = proxy([[0, 0, 0, 1], [0, 0, 0, 1]])
        sensor = term._env.scene["target_frame"]
        sensor.data.target_pos_w = proxy([[[0, 0, 0]], [[0, 0, 0]]])
        axis = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])

        def rotations(degrees):
            return math_utils.quat_from_angle_axis(torch.deg2rad(torch.tensor(degrees)), axis)

        sensor.data.target_quat_w = NS(torch=rotations([100.0, 80.0])[:, None])
        term._get_ee_in_base_frame = lambda ids: (
            term._asset.data.root_pos_w.torch[ids],
            term._asset.data.root_quat_w.torch[ids],
        )
        cfg = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=True)
        handler = _GoToFrameHandler(cfg, term)
        ids = torch.tensor([0, 1])
        handler.reset(ids)
        sensor.data.target_quat_w.torch[:] = rotations([80.0, 100.0])[:, None]
        _, actual = handler.get_target_in_base_frame(ids)
        expected = rotations([-100.0, 100.0])
        torch.testing.assert_close(math_utils.quat_error_magnitude(actual, expected), torch.zeros(2), atol=1e-6, rtol=0)
        handler.reset(torch.tensor([0]))
        _, actual = handler.get_target_in_base_frame(ids)
        expected = rotations([80.0, 100.0])
        torch.testing.assert_close(math_utils.quat_error_magnitude(actual, expected), torch.zeros(2), atol=1e-6, rtol=0)

    def test_reset_refreshes_native_frame_buffers_without_stepping_physics(self):
        # FK changes the body state, but the frame sensor reads a separate
        # native buffer. Both layers must be updated after every reset.
        state = torch.zeros(1, 7)
        native_buffer = torch.zeros_like(state)
        frame_buffer = torch.zeros_like(state)
        reset_pose = torch.tensor([[0.8, 0.1, 0.7, 0, 0, 0, 1.0]])
        frame = NS(
            cfg=NS(target_frames=["goal"]),
            update=lambda dt, force_recompute: frame_buffer.copy_(native_buffer),
        )
        physics = NS(
            forward=lambda: state.copy_(reset_pose),
            get_state=lambda: state,
            _newton_frame_transform_sensors=[NS(update=lambda current: native_buffer.copy_(current))],
        )
        env = NS(scene=NS(sensors={"target_frame": frame}), sim=NS(physics_manager=physics))
        refresh_frame_sensors(env)
        torch.testing.assert_close(frame_buffer, reset_pose)
        reset_pose[:, 0] = 1.2
        refresh_frame_sensors(env)
        torch.testing.assert_close(frame_buffer, reset_pose)

    def test_unsampled_reference_is_rejected_instead_of_targeting_world_origin(self):
        term = geometry_term()
        term._env.scene["target_frame"].data.target_quat_w = proxy([[[0, 0, 0, 0]]])
        cfg = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=False)
        initial = (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1]]))
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

    def test_editing_angle_deg_disables_valve_angle_and_caption_names_the_source(self):
        cfg = RotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            use_valve_angle=True,
            angle_deg=90.0,
        )
        self.assertTrue(apply_authored_command_field(cfg, "angle_deg", -45.0))
        self.assertEqual(cfg.angle_deg, -45.0)
        self.assertFalse(cfg.use_valve_angle)
        self.assertFalse(apply_authored_command_field(cfg, "angle_deg", -90.0))
        handler = NS(angle_rad_tensor=torch.tensor([-math.pi / 2]))
        self.assertEqual(
            effective_arc_caption(cfg, handler),
            "Effective arc: **-90.0°** (from Angle (deg))",
        )
        cfg.use_valve_angle = True
        cfg.angle_deg = 90.0
        self.assertIn("Angle (deg)=90 is unused", effective_arc_caption(cfg, handler))
        cfg.max_ee_rotation_deg = 60.0
        cfg.use_valve_angle = False
        self.assertIn("clamped to ±60°", effective_arc_caption(cfg, handler))

    def test_segment_samples_curobo_preview_plan(self):
        cfg = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=False)
        start = (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1]]))
        end = (torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        segment = Segment(cfg, start, end, 1.0)
        cfg._preview_plan = {
            "joints": torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
            "joint_names": ["arm_sh0", "arm_sh1"],
            "tcp_pos_b": torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
            "tcp_quat_b": torch.tensor([[0.0, 0, 0, 1], [0.0, 0, 0, 1]]),
        }
        pos, _ = segment.sample(0.5)
        torch.testing.assert_close(pos[0], torch.tensor([0.2, 0.0, 0.0]))
        joints, names = segment.sample_joints(0.5)
        torch.testing.assert_close(joints, torch.tensor([0.5, 1.0]))
        self.assertEqual(names, ["arm_sh0", "arm_sh1"])

    def test_stale_preview_plan_disconnects_until_later_legs_are_invalidated(self):
        identity = torch.tensor([[0.0, 0, 0, 1]])
        first = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=False)
        second = GoToFrameCfg(frame_name="target_frame", target_frame_name="goal", canonicalize_upward=False)
        first_seg = Segment(first, (torch.zeros(1, 3), identity), (torch.tensor([[0.4, 0, 0]]), identity), 0.16)
        second_seg = Segment(second, (torch.tensor([[0.4, 0, 0]]), identity), (torch.tensor([[0.8, 0, 0]]), identity), 0.16)
        second._preview_plan = {
            "joints": torch.zeros(2, 1),
            "joint_names": ["arm_sh0"],
            "tcp_pos_b": torch.tensor([[0.1, 0.0, 0.0], [0.8, 0.0, 0.0]]),
            "tcp_quat_b": torch.tensor([[0.0, 0, 0, 1], [0.0, 0, 0, 1]]),
        }
        first_seg.end = (torch.tensor([[0.55, 0, 0]]), identity)
        second_seg.start = first_seg.end
        gap = torch.linalg.vector_norm(second_seg.sample(0.0)[0] - first_seg.end[0])
        self.assertGreater(float(gap), 0.2)
        invalidate_preview_plans_from([first, second], 1)
        torch.testing.assert_close(second_seg.sample(0.0)[0], first_seg.end[0])
        pos, _, _ = concat_segment_tcp([first_seg, second_seg], dt=0.05)
        self.assertFalse(bool(torch.any(torch.abs(pos[:, 0] - 0.1) < 1e-3)))

    def test_concat_ignores_cached_plan_so_replanning_uses_live_geometry(self):
        identity = torch.tensor([[0.0, 0, 0, 1]])
        start = (torch.zeros(1, 3), identity)
        mid = (torch.tensor([[0.4, 0, 0]]), identity)
        end = (torch.tensor([[0.8, 0, 0]]), identity)
        first = Segment(GoToFrameCfg(frame_name="target_frame", target_frame_name="goal"), start, mid, 0.16)
        second = Segment(GoToFrameCfg(frame_name="target_frame", target_frame_name="goal"), mid, end, 0.16)
        second.cfg._preview_plan = {
            "joints": torch.zeros(2, 1),
            "joint_names": ["arm_sh0"],
            "tcp_pos_b": torch.tensor([[0.0, 1.0, 0.0], [0.8, 1.0, 0.0]]),
            "tcp_quat_b": torch.tensor([[0.0, 0, 0, 1], [0.0, 0, 0, 1]]),
        }
        pos, _, _ = concat_segment_tcp([first, second], dt=0.05)
        self.assertLess(float(torch.max(torch.abs(pos[:, 1]))), 1e-5)

    def test_concat_segment_tcp_drops_duplicate_endpoints(self):
        start = (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1]]))
        mid = (torch.tensor([[0.4, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        end = (torch.tensor([[0.8, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        first = Segment(GoToFrameCfg(frame_name="target_frame", target_frame_name="goal"), start, mid, 0.16)
        second = Segment(GoToFrameCfg(frame_name="target_frame", target_frame_name="goal"), mid, end, 0.16)
        hold = Segment(GripperCommand(open_gripper=False, duration_s=0.3), end, end, 0.3)
        pos, _, ranges = concat_segment_tcp([first, second, hold], dt=0.05)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[1][0], ranges[0][1])
        self.assertEqual(ranges[2][1] - ranges[2][0], 1)
        self.assertEqual(int(pos.shape[0]), ranges[-1][1])

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
        self.assertEqual(ik.last_debug["jacobian_body"], 0)
        self.assertLess(ik.last_debug["pos_error_mm"], 1.0)


class ReplayPathTests(unittest.TestCase):
    def test_goto_draws_remaining_motion_without_advancing_handler(self):
        term = geometry_term()
        term._env.scene["target_frame"].data.target_pos_w = proxy([[[1, 0, 0]]])
        term._env.scene["target_frame"].data.target_quat_w = proxy([[[0, 0, 1, 0]]])
        initial = (torch.zeros(1, 3), torch.tensor([[0.0, 0, 0, 1]]))
        cfg = GoToFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            canonicalize_upward=False,
            velocity=1.0,
            angular_velocity=1.0,
        )
        handler = build_segments(term, [cfg], initial)[0].handler
        before = handler.command_pos_b.clone()
        command = torch.tensor([[1.0, 0.4, 0, 0, 0, 0, 0, 1]])
        pos, quat, index = active_command_path(handler, command)
        self.assertEqual(index, 0)
        torch.testing.assert_close(pos[0], command[0, 1:4])
        torch.testing.assert_close(pos[-1], torch.tensor([1.0, 0, 0]))
        # Translation finishes before the slower orientation change.
        torch.testing.assert_close(pos[16], pos[-1])
        torch.testing.assert_close(quat[-1].abs(), torch.tensor([0.0, 0, 1, 0]), atol=1e-6, rtol=0)
        torch.testing.assert_close(handler.command_pos_b, before)

    def test_signed_screw_path_and_progress_marker(self):
        term = geometry_term()
        initial = (torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 0, 0, 1]]))
        cfg = ScrewFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            axis=(0, 0, 1),
            use_valve_angle=False,
            angle_deg=-180,
            axial_distance=0.4,
        )
        handler = build_segments(term, [cfg], initial)[0].handler
        handler._progress_abs[0] = math.pi / 2
        command = torch.tensor([[-1.0, 0, -1, 0.2, 0, 0, 0, 1]])
        pos, quat, index = active_command_path(handler, command)
        self.assertEqual(index, 16)
        torch.testing.assert_close(pos[index], torch.tensor([0.0, -1, 0.2]), atol=1e-6, rtol=0)
        torch.testing.assert_close(pos[-1], torch.tensor([-1.0, 0, 0.4]), atol=1e-6, rtol=0)
        torch.testing.assert_close(quat[index], torch.tensor([0.0, 0, -math.sqrt(0.5), math.sqrt(0.5)]))
        self.assertAlmostEqual(float(handler._progress_abs[0]), math.pi / 2, places=6)

    def test_actual_curobo_path_takes_precedence_and_gripper_has_no_path(self):
        term = geometry_term()
        cfg = CuroboPlannedGoToFrameCfg(
            frame_name="target_frame", target_frame_name="goal", robot_joint_names=["joint"]
        )
        handler = cfg.class_type(cfg, term)
        handler._waypoint_pos_b = torch.tensor([[[0.0, 0, 0], [0.5, 0.4, 0.3], [1, 0, 0]]])
        handler._waypoint_quat_b = torch.tensor([[[0.0, 0, 0, 1]] * 3])
        handler._waypoint_index[0] = 3  # Completion sentinel stays within the marker array.
        command = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 1]])
        pos, quat, index = active_command_path(handler, command)
        self.assertEqual(index, 2)
        torch.testing.assert_close(pos, handler._waypoint_pos_b[0])
        torch.testing.assert_close(quat, handler._waypoint_quat_b[0])
        self.assertIsNone(active_command_path(_GripperHandler(GripperCommand(), term), command))


class RotateGraspPlaneTests(unittest.TestCase):
    """The TCP must orbit in the grasp plane, not snap onto the hub first."""

    def _handler(self, initial, *, hub=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), angle_deg=-90.0, cls=RotateFrameCfg):
        term = geometry_term()
        term._env.scene["target_frame"].data.target_pos_w = proxy([[list(hub)]])
        cfg = cls(
            frame_name="target_frame",
            target_frame_name="goal",
            axis=axis,
            angle_deg=angle_deg,
            use_valve_angle=False,
            angular_velocity=0.3,
        )
        return build_segments(term, [cfg], initial)[0]

    def test_angle_zero_stays_at_grasp_not_hub_plane(self):
        initial = (torch.tensor([[1.0, 0.0, 0.05]]), torch.tensor([[0.0, 0, 0, 1]]))
        segment = self._handler(initial)
        handler = segment.handler
        torch.testing.assert_close(handler.axial_vec, torch.tensor([[0.0, 0.0, 0.05]]))
        torch.testing.assert_close(handler.radius_vec, torch.tensor([[1.0, 0.0, 0.0]]))
        handler.angle_rad_tensor[0] = 0.0
        pos, _ = handler.get_target_in_base_frame(torch.tensor([0]))
        torch.testing.assert_close(pos, initial[0], atol=1e-6, rtol=0)
        pos0, _ = segment.sample(0.0)
        torch.testing.assert_close(pos0, initial[0], atol=1e-6, rtol=0)

    def test_orbit_keeps_stem_offset(self):
        initial = (torch.tensor([[1.0, 0.0, 0.05]]), torch.tensor([[0.0, 0, 0, 1]]))
        segment = self._handler(initial, angle_deg=-90)
        pos, _ = segment.handler.get_target_in_base_frame(torch.tensor([0]))
        torch.testing.assert_close(pos, torch.tensor([[0.0, -1.0, 0.05]]), atol=1e-6, rtol=0)
        pos_half, _ = segment.sample(0.5)
        torch.testing.assert_close(pos_half, torch.tensor([[0.7071068, -0.7071068, 0.05]]), atol=1e-5, rtol=0)

    def test_first_update_does_not_slide_along_axis(self):
        initial = (torch.tensor([[1.0, 0.0, 0.05]]), torch.tensor([[0.0, 0, 0, 1]]))
        handler = self._handler(initial).handler
        command = handler.update(torch.tensor([True]))
        # 0.3 rad/s * 0.05 s is a tiny yaw; Z must not jump from 0.05 to the hub.
        self.assertAlmostEqual(float(command[0, 3]), 0.05, places=5)
        self.assertGreater(float(command[0, 1]), 0.99)

    def test_small_valve_like_stem_offset_does_not_retract_to_hub(self):
        # Same geometry as the play dump: hub 7 mm further along +X than the TCP.
        hub = (0.957, 0.0, 0.0)
        tcp = torch.tensor([[0.950, 0.0237, 0.0]])
        initial = (tcp, torch.tensor([[0.0, 0, 0, 1]]))
        handler = self._handler(initial, hub=hub, axis=(1.0, 0.0, 0.0), angle_deg=-90).handler
        saved_angle = float(handler.angle_rad_tensor[0])
        handler.angle_rad_tensor[0] = 0.0
        pos0, _ = handler.get_target_in_base_frame(torch.tensor([0]))
        torch.testing.assert_close(pos0, tcp, atol=1e-6, rtol=0)
        handler.angle_rad_tensor[0] = saved_angle
        command = handler.update(torch.ones(1, dtype=torch.bool))
        # Sliding onto the hub would move X from 0.950 to 0.957 on the first tick.
        self.assertAlmostEqual(float(command[0, 1]), 0.950, places=4)
        path, _, _ = active_command_path(handler, command)
        torch.testing.assert_close(path[0], tcp[0], atol=1e-5, rtol=0)
        torch.testing.assert_close(path[:, 0], torch.full((path.shape[0],), 0.950), atol=1e-5, rtol=0)

    def test_screw_keeps_grasp_offset_and_adds_pitch(self):
        initial = (torch.tensor([[1.0, 0.0, 0.05]]), torch.tensor([[0.0, 0, 0, 1]]))
        segment = self._handler(initial, angle_deg=-180, cls=ScrewFrameCfg)
        segment.cfg.axial_distance = 0.4
        pos, _ = segment.sample(0.5)
        torch.testing.assert_close(pos, torch.tensor([[0.0, -1.0, 0.25]]), atol=1e-6, rtol=0)

    def test_max_ee_rotation_clamps_commanded_arc(self):
        initial = (torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0, 0, 1]]))
        term = geometry_term()
        cfg = RotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            axis=(0.0, 0.0, 1.0),
            angle_deg=-180.0,
            use_valve_angle=False,
            max_ee_rotation_deg=90.0,
        )
        handler = build_segments(term, [cfg], initial)[0].handler
        self.assertAlmostEqual(float(handler.angle_rad_tensor[0]), -math.pi / 2, places=6)
        cfg.max_ee_rotation_deg = 0.0
        handler = build_segments(term, [cfg], initial)[0].handler
        self.assertAlmostEqual(float(handler.angle_rad_tensor[0]), -math.pi, places=6)

    def test_discontinuous_curobo_arc_raises_instead_of_falling_back(self):
        term = geometry_term()
        cfg = CuroboPlannedRotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            robot_joint_names=["arm_sh0", "arm_el0"],
            use_valve_angle=False,
            max_joint_step=0.15,
        )
        handler = _CuroboPlannedRotateFrameHandler(cfg, term)
        waypoints = torch.zeros(1, 3, 2)
        waypoints[0, 1, 0] = 0.4
        with self.assertRaisesRegex(RuntimeError, r"infeasible: max joint step=0\.400 rad"):
            handler._raise_if_joint_discontinuity(waypoints, ["arm_sh0", "arm_el0"])

    def test_infeasible_arc_dump_includes_robot_state_and_requested_plan(self):
        term = geometry_term()
        robot = term._asset
        robot.find_joints = lambda names, preserve_order=True: ([0, 1], list(names))
        robot.data.joint_pos = proxy([[0.11, -0.22]])
        robot.data.joint_pos_limits = proxy([[[-1.0, 1.0], [-2.0, 2.0]]])
        cfg = CuroboPlannedRotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            robot_joint_names=["arm_sh0", "arm_wr1"],
            use_valve_angle=False,
            angle_deg=-90.0,
            max_joint_step=0.15,
        )
        handler = _CuroboPlannedRotateFrameHandler(cfg, term)
        handler.angle_rad_tensor[:] = -math.pi / 2
        handler.axis_pos_b[:] = torch.tensor([0.5, 0.0, 0.4])
        handler.rot_axis_b[:] = torch.tensor([-1.0, 0.0, 0.0])
        handler.radius_vec[:] = torch.tensor([0.0, 0.08, 0.0])
        handler.axial_vec[:] = torch.tensor([0.01, 0.0, 0.0])
        n_wp = 5
        waypoints = torch.zeros(1, n_wp, 2)
        waypoints[0, :, 0] = torch.tensor([0.11, 0.32, 0.32, 0.32, 0.32])
        waypoints[0, :, 1] = torch.tensor([0.0, 0.05, 0.10, 0.45, 0.50])
        tcp_pos = torch.zeros(n_wp, 3)
        tcp_pos[:, 0] = 0.51
        tcp_pos[:, 1] = torch.linspace(0.08, -0.08, n_wp)
        tcp_pos[:, 2] = 0.4
        tcp_quat = torch.zeros(n_wp, 4)
        tcp_quat[:, 3] = 1.0
        angles = torch.linspace(0.0, -math.pi / 2, n_wp)
        with self.assertRaises(RuntimeError) as caught:
            handler._raise_if_joint_discontinuity(
                waypoints,
                ["arm_sh0", "arm_wr1"],
                tcp_pos_b=tcp_pos,
                tcp_quat_b=tcp_quat,
                angles=angles,
            )
        dump = str(caught.exception)
        self.assertIn("waypoint 2->3, joint arm_wr1", dump)
        self.assertIn("robot state (grasp seed / waypoint 0)", dump)
        self.assertIn("arm_sh0=0.1100", dump)
        self.assertIn("measured:", dump)
        self.assertIn("arm_wr1=-0.2200", dump)
        self.assertIn("limits:", dump)
        self.assertIn("requested arc", dump)
        self.assertIn("angle=-90.00 deg", dump)
        self.assertIn("radius=0.0800 m", dump)
        self.assertIn("axial_offset=0.0100 m", dump)
        self.assertIn("arm_wr1=0.350@2->3 <<", dump)
        self.assertIn("all joints exceeding max_joint_step=0.150 rad", dump)
        self.assertIn("arm_sh0 0.210 rad at 0->1", dump)
        self.assertIn("arm_wr1 0.350 rad at 2->3 << worst", dump)
        self.assertIn("before limits:", dump)
        self.assertIn("after  limits:", dump)
        self.assertIn("tcp step:", dump)
        self.assertIn("[0]", dump)
        self.assertIn("[2]", dump)
        self.assertIn("[3]", dump)
        self.assertIn("<< jump from 0 on arm_sh0", dump)
        self.assertIn("<< jump from 2 on arm_wr1", dump)

    def test_infeasible_rotation_message_tells_user_to_reduce_the_arc(self):
        err = RuntimeError(
            "cuRobo valve arc is infeasible: "
            "max joint step=0.338 rad > max_joint_step=0.150 rad "
            "(waypoint 134->135, joint arm_wr1)\n--- robot state ---"
        )
        handler = NS(angle_rad_tensor=torch.tensor([-math.pi]))
        message = format_infeasible_rotation_message(err, handler)
        self.assertTrue(message.startswith("Rotation is infeasible."))
        self.assertIn("waypoint 134->135, joint arm_wr1", message)
        self.assertIn("Commanded arc is -180.0°", message)
        self.assertIn("max_ee_rotation_deg", message)
        self.assertNotIn("--- robot state ---", message)
        status = format_editor_plan_failure(RuntimeError(message))
        self.assertIn("**Rotation is infeasible**", status)
        self.assertIn("Reduce the rotation", status)

    def test_editor_rotate_plan_raises_instead_of_dls_fallback(self):
        term = geometry_term()
        cfg = CuroboPlannedRotateFrameCfg(
            frame_name="target_frame",
            target_frame_name="goal",
            robot_joint_names=["arm_sh0", "arm_wr1"],
            use_valve_angle=False,
            angle_deg=-180.0,
            max_joint_step=0.15,
        )
        identity = torch.tensor([[0.0, 0, 0, 1]])
        segment = Segment(cfg, (torch.zeros(1, 3), identity), (torch.zeros(1, 3), identity), 1.0)
        solved = {"dls": False}

        class StubIK:
            def reset(self):
                return None

            def write(self, q):
                return None

            def solve(self, *args, **kwargs):
                solved["dls"] = True
                raise AssertionError("infeasible rotate must not fall back to DLS")

            def apply_named_joints(self, *args, **kwargs):
                return None

            def _gripper_q(self, q, gripper_open):
                return q

            robot = term._asset

        class FakeHandler:
            def __init__(self, cfg, command_term):
                self.cfg = cfg
                self.angle_rad_tensor = torch.tensor([-math.pi])

            def reset(self, env_ids):
                return None

            def _plan(self, env_ids):
                raise RuntimeError(
                    "cuRobo valve arc is infeasible: "
                    "max joint step=0.338 rad > max_joint_step=0.150 rad "
                    "(waypoint 134->135, joint arm_wr1)"
                )

        with patch(
            "isaaclab_hiveboard.utils.command_preview._CuroboPlannedRotateFrameHandler",
            FakeHandler,
        ):
            with self.assertRaisesRegex(RuntimeError, r"Rotation is infeasible"):
                _plan_curobo_per_segment(term, [segment], StubIK())
        self.assertFalse(solved["dls"])
        self.assertIsNone(cfg._preview_plan)


if __name__ == "__main__":
    unittest.main()
