# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constants copied from the HiveBoard website Spot demo.

Source: ``dependencies/hiveboard-bench.github.io`` (``spot.xml``, ``robots.json``,
``tools/build-sim-assets.py``). Isaac Lab stores quaternions as ``xyzw``.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_hiveboard.assets.spot.constants import ARM_ARMATURE, ARM_EFFORT_LIMIT
from isaaclab_hiveboard.assets.spot.spot import SPOT_ARM_UUC_USD

# Website ``<option timestep="0.002"/>`` and trajectory sample rate.
PHYSICS_DT = 1.0 / 500.0
TRAJ_RATE_HZ = 50
DECIMATION_HZ = 50
DECIMATION = max(1, round((1.0 / DECIMATION_HZ) / PHYSICS_DT))

# Per-joint explicit IdealPD (4x hardware-style kp/kd, except the gripper).
# Gripper is a tuned open/close baseline (was 16.0 * 4 / 0.32 * 4, which rang
# ~7 Hz on binary steps); see logs/gain_opt/2026-09-11_22-34-45. Effort/armature match the URDF.
ARM_STIFFNESS: tuple[float, ...] = (
    120.0 * 4,
    120.0 * 4,
    120.0 * 4,
    100.0 * 4,
    100.0 * 4,
    100.0 * 4,
    30.5,
)
ARM_DAMPING: tuple[float, ...] = (
    2.0 * 4,
    2.0 * 4,
    2.0 * 4,
    2.0 * 4,
    2.0 * 4,
    2.0 * 4,
    0.35,
)

# Website home: arm ``[0, -1.9, 2.0, 0, -0.6, 0]``, gripper ``-1.5``.
# Body is welded at z=0.462321. Every Spot body has MuJoCo ``gravcomp=1``.
BODY_POS: tuple[float, float, float] = (0.0, 0.0, 0.462321)
HOME_ARM: tuple[float, ...] = (0.0, -1.9, 2.0, 0.0, -0.6, 0.0)
HOME_GRIP = -1.5

# Upright honeycomb: MuJoCo wxyz ``(0, 0, 0, 1)`` = 180 deg yaw.
BOARD_POS: tuple[float, float, float] = (1.08, 0.0, 0.70)
BOARD_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.0)

# ``valve_valvula_esfera`` local pose composed through the 180 deg board yaw:
# local pos (0.0400277, 0.043301, 0.075), local wxyz (0.86619, 0.499715, 0, 0).
VALVE_POS: tuple[float, float, float] = (1.0399723, -0.043301, 0.775)
VALVE_QUAT_XYZW: tuple[float, float, float, float] = (0.0, 0.499715, 0.86619, 0.0)

# Website valve joint: damping 0.1, armature 0.002, frictionloss 0.02, range [-pi/2, 0].
VALVE_DAMPING = 0.1
VALVE_ARMATURE = 0.002
VALVE_FRICTION = 0.02
VALVE_JOINT_CLOSED = 0.0
VALVE_JOINT_OPEN = -math.pi / 2
VALVE_GOAL = VALVE_JOINT_OPEN

# Stand boxes from ``spot.xml`` (visual only).
STAND_POST_POS: tuple[float, float, float] = (1.11, 0.0, 0.35)
STAND_POST_SIZE: tuple[float, float, float] = (0.032, 0.032, 0.70)
STAND_FOOT_POS: tuple[float, float, float] = (1.11, 0.0, 0.01)
STAND_FOOT_SIZE: tuple[float, float, float] = (0.34, 0.34, 0.02)

TRAJECTORY_JSON = Path(__file__).with_name("trajectories") / "spot_bench_valve.json"
# Gains validation clip: same arm motion, gripper snapped to open/close only.
GAINS_TRAJECTORY_JSON = Path(__file__).with_name("trajectories") / "spot_bench_gains.json"
# Website ``<site name="tcp">`` on ``arm_link_wr1`` (``spot.xml``). Key beads use this frame.
TCP_SITE_POS: tuple[float, float, float] = (0.198984, 0.000566814, -0.037663)
# MuJoCo wxyz ``(0.448024, 0.546678, 0.542338, 0.454189)`` as Isaac Lab xyzw.
TCP_SITE_QUAT_XYZW: tuple[float, float, float, float] = (0.546678, 0.542338, 0.454189, 0.448024)
# Upright board face: modules stick out −X, so approach into the board is +X.
BOARD_INTO: tuple[float, float, float] = (1.0, 0.0, 0.0)

BENCH_JOINT_POS: dict[str, float] = {
    "arm_sh0": HOME_ARM[0],
    "arm_sh1": HOME_ARM[1],
    "arm_el0": HOME_ARM[2],
    "arm_el1": HOME_ARM[3],
    "arm_wr0": HOME_ARM[4],
    "arm_wr1": HOME_ARM[5],
    "arm_f1x": HOME_GRIP,
    "fl_hx": 0.12,
    "fr_hx": -0.12,
    "hl_hx": 0.12,
    "hr_hx": -0.12,
    "fl_hy": 0.5,
    "fr_hy": 0.5,
    "hl_hy": 0.5,
    "hr_hy": 0.5,
    "fl_kn": -1.0,
    "fr_kn": -1.0,
    "hl_kn": -1.0,
    "hr_kn": -1.0,
}

ARM_JOINT_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "arm_f1x",
]
ARM_ACTUATOR_NAMES = [
    "arm_sh0",
    "arm_sh1",
    "arm_el0",
    "arm_el1",
    "arm_wr0",
    "arm_wr1",
    "gripper",
]


def _arm_actuator(name: str, effort_index: int) -> IdealPDActuatorCfg:
    return IdealPDActuatorCfg(
        joint_names_expr=[name],
        effort_limit=ARM_EFFORT_LIMIT[effort_index],
        stiffness=ARM_STIFFNESS[effort_index],
        damping=ARM_DAMPING[effort_index],
        armature=ARM_ARMATURE[effort_index],
    )


SPOT_ARM_BENCH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SPOT_ARM_UUC_USD,
        activate_contact_sensors=True,
        # PhysX: disable_gravity on the root (nested-stop). Newton: per-body
        # mjc:gravcomp is applied at startup on every robot link.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # USD already has a root FixedJoint; do not set fix_root_link (PhysX-only writer).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=BODY_POS,
        joint_pos=BENCH_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fixed_legs": IdealPDActuatorCfg(
            joint_names_expr=["[fh][lr]_(hx|hy|kn)"],
            effort_limit=90.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.02,
            armature=0.01,
        ),
        "arm_sh0": _arm_actuator("arm_sh0", 0),
        "arm_sh1": _arm_actuator("arm_sh1", 1),
        "arm_el0": _arm_actuator("arm_el0", 2),
        "arm_el1": _arm_actuator("arm_el1", 3),
        "arm_wr0": _arm_actuator("arm_wr0", 4),
        "arm_wr1": _arm_actuator("arm_wr1", 5),
        "gripper": _arm_actuator("arm_f1x", 6),
    },
)

# Robot-only gain env shares the same explicit IdealPD layout.
SPOT_ARM_GAINS_CFG = SPOT_ARM_BENCH_CFG


def apply_spot_arm_gains(robot, stiffness, damping) -> None:
    """Write per-joint arm/gripper IdealPD gains onto a live Spot articulation.

    ``stiffness`` / ``damping`` are length-7 (broadcast to every env) or
    ``(num_envs, 7)`` so a vectorized rollout can score many gain sets at once.
    """
    import numpy as np
    import torch

    stiffness = np.asarray(stiffness, dtype=np.float32)
    damping = np.asarray(damping, dtype=np.float32)
    n_env = int(robot.num_instances)
    if stiffness.ndim == 1:
        stiffness = np.repeat(stiffness[None, :], n_env, axis=0)
    if damping.ndim == 1:
        damping = np.repeat(damping[None, :], n_env, axis=0)
    if stiffness.shape != (n_env, len(ARM_ACTUATOR_NAMES)) or damping.shape != (
        n_env,
        len(ARM_ACTUATOR_NAMES),
    ):
        raise ValueError(
            f"Expected stiffness/damping shape ({n_env}, {len(ARM_ACTUATOR_NAMES)}), "
            f"got {stiffness.shape}/{damping.shape}."
        )
    for index, name in enumerate(ARM_ACTUATOR_NAMES):
        actuator = robot.actuators[name]
        kp_t = torch.as_tensor(stiffness[:, index], device=robot.device, dtype=torch.float32)
        kd_t = torch.as_tensor(damping[:, index], device=robot.device, dtype=torch.float32)
        actuator.stiffness[:] = kp_t.reshape(actuator.stiffness.shape)
        actuator.damping[:] = kd_t.reshape(actuator.damping.shape)
