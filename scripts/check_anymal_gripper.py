# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep the baked 2F-140 through open/close commands using Newton's solver.

    uv run python scripts/check_anymal_gripper.py
"""

from __future__ import annotations

import argparse

import newton
import numpy as np
import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from generate_anymal_newton_usd import ASSEMBLY_USDA, quat_to_mat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    wp.init()
    wp.set_device(args.device)
    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/Gripper", "Xform")
    root.GetReferences().AddReference(str(ASSEMBLY_USDA), "/anymal_dynaarm_robotiq/robotiq_2f_140")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    palm = root.GetChild("robotiq_base_link")
    palm_world = UsdGeom.XformCache().GetLocalToWorldTransform(palm)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Gripper/fixed_to_world")
    fixed.CreateBody1Rel().SetTargets([palm.GetPath()])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(palm_world.ExtractTranslation()))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(palm_world.ExtractRotationQuat()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    builder = newton.ModelBuilder(gravity=0.0)
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(stage, enable_self_collisions=False, skip_mesh_approximation=True)
    assert len(builder.equality_constraint_label) == 5, "Missing gripper loop/joint constraints"
    assert len(builder.joint_q) == 8, "All eight gripper hinges must remain mobile"
    motor = builder.joint_label.index("/Gripper/finger_joint")
    motor_dof = builder.joint_qd_start[motor]
    for i in range(len(builder.joint_target_ke)):
        builder.joint_target_ke[i] = 0.0
        builder.joint_target_kd[i] = 0.0
    model = builder.finalize()
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=args.device == "cpu", integrator="implicitfast")
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    control = model.control()
    left = model.body_label.index("/Gripper/left_inner_finger")
    right = model.body_label.index("/Gripper/right_inner_finger")
    initial_rot = quat_to_mat(state.body_q.numpy()[left, 3:])
    dt = 0.002
    worst_parallel_error = 0.0
    for target in (0.0, 0.35, 0.7, 0.0):
        for step in range(2000):
            q = state.joint_q.numpy()
            qd = state.joint_qd.numpy()
            force = -0.1 * qd
            force[motor_dof] = np.clip(80.0 * (target - q[builder.joint_q_start[motor]]) - 4.0 * qd[motor_dof], -40, 40)
            control.joint_f.assign(force)
            state.clear_forces()
            solver.step(state, next_state, control, None, dt)
            state, next_state = next_state, state
            if step % 10 == 0:
                poses = state.body_q.numpy()
                left_rot, right_rot = (quat_to_mat(poses[body, 3:]) for body in (left, right))
                # Frobenius norm of the rotation difference gives the angle
                # between the pads, independent of the hand's world pose.
                error = np.degrees(2 * np.arcsin(np.clip(np.linalg.norm(left_rot - right_rot) / np.sqrt(8), 0, 1)))
                worst_parallel_error = max(worst_parallel_error, float(error))
                assert error < 2.0, f"Fingers lost parallel alignment during motion: {error:.3f} degrees"
        poses = state.body_q.numpy()
        left_rot = quat_to_mat(poses[left, 3:])
        right_rot = quat_to_mat(poses[right, 3:])
        position = state.joint_q.numpy()[builder.joint_q_start[motor]]
        print(
            f"[GRIPPER] target={target:.2f}, motor={position:.6f}, "
            f"worst alignment error={worst_parallel_error:.4f} deg"
        )
        np.testing.assert_allclose(left_rot, right_rot, atol=2e-3)
        np.testing.assert_allclose(left_rot, initial_rot, atol=2e-3)
        np.testing.assert_allclose(position, target, atol=2e-3)
        assert np.isfinite(poses).all()
    print("[OK] Robotiq fingertips stay parallel throughout the open/close sweep")


if __name__ == "__main__":
    main()
