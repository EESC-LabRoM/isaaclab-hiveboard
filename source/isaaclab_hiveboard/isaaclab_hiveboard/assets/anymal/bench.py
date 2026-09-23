# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal-D bench constants sharing the Spot website board/valve placement.

The board, stand, and ball-valve poses are identical to
:mod:`isaaclab_hiveboard.assets.spot.bench` so the traj_edit Cartesian beads
(TCP positions in the env frame) can be shared between the Spot and ANYmal
cuRobo valve envs. Only the robot base, home configuration, and flange→TCP
offset differ.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_hiveboard.assets.anymal import (
    ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG,
    DYNAARM_JOINT_NAMES,
    ROBOTIQ_INIT_JOINT_POS,
)
from isaaclab_hiveboard.assets.end_effector import ANYMAL_EE
from isaaclab_hiveboard.assets.spot.bench import (
    BOARD_POS,
    BOARD_QUAT_XYZW,
    DECIMATION,
    PHYSICS_DT,
    STAND_FOOT_POS,
    STAND_FOOT_SIZE,
    STAND_POST_POS,
    STAND_POST_SIZE,
    VALVE_ARMATURE,
    VALVE_DAMPING,
    VALVE_FRICTION,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
    VALVE_POS,
    VALVE_QUAT_XYZW,
)

__all__ = [
    "BOARD_POS",
    "BOARD_QUAT_XYZW",
    "DECIMATION",
    "PHYSICS_DT",
    "STAND_FOOT_POS",
    "STAND_FOOT_SIZE",
    "STAND_POST_POS",
    "STAND_POST_SIZE",
    "VALVE_ARMATURE",
    "VALVE_DAMPING",
    "VALVE_FRICTION",
    "VALVE_JOINT_CLOSED",
    "VALVE_JOINT_OPEN",
    "VALVE_POS",
    "VALVE_QUAT_XYZW",
    "ANYMAL_ARM_JOINT_NAMES",
    "ANYMAL_BASE_POS",
    "ANYMAL_HOME_ARM",
    "FLANGE_TO_TCP_POS",
    "FLANGE_TO_TCP_QUAT_XYZW",
    "ANYMAL_ARM_BENCH_CFG",
    "ANYMAL_BAKED_USDA",
    "ANYMAL_ARM_NEWTON_CFG",
    "ANYMAL_NEWTON_JOINT_POS",
    "ANYMAL_NEWTON_GRIPPER_OPEN",
    "ANYMAL_NEWTON_GRIPPER_CLOSE",
    "NEWTON_GRIPPER_JOINT_NAMES",
]

NEWTON_GRIPPER_JOINT_NAMES = ["finger_joint"]
"""The motor joint; USD constraints drive the parallel finger linkage."""

# Fixed-base standing height from the ANYmal ball-valve env (legs PD-held).
ANYMAL_BASE_POS: tuple[float, float, float] = (0.0, 0.0, 0.6)
# Forward-reach manipulation pose (same as the ANYmal ball-valve scene).
ANYMAL_HOME_ARM: tuple[float, ...] = (0.0, 0.0, 1.7, 0.0, 0.0, 1.5708)

ANYMAL_ARM_JOINT_NAMES = list(DYNAARM_JOINT_NAMES)

# Flange → canonical TCP, MEASURED from the baked assembly (see `report_tcp_offset`
# in scripts/generate_anymal_newton_usd.py; re-measure after any re-bake that
# moves the weld). With the 180 deg mount yaw the weld twist cancels and this
# currently coincides with the ANYMAL_EE palm→TCP profile.
FLANGE_TO_TCP_POS = (0.0, 0.0, 0.20)
FLANGE_TO_TCP_QUAT_XYZW = (0.0, -0.7071068, 0.0, 0.7071068)

ANYMAL_BENCH_JOINT_POS: dict[str, float] = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
    "dynaarm_shoulder_rotation": ANYMAL_HOME_ARM[0],
    "dynaarm_shoulder_flexion": ANYMAL_HOME_ARM[1],
    "dynaarm_elbow_flexion": ANYMAL_HOME_ARM[2],
    "dynaarm_forearm_rotation": ANYMAL_HOME_ARM[3],
    "dynaarm_wrist_flexion": ANYMAL_HOME_ARM[4],
    "dynaarm_wrist_rotation": ANYMAL_HOME_ARM[5],
    **ROBOTIQ_INIT_JOINT_POS,
}

# All eight tree joints start at the linkage's open configuration.
ANYMAL_NEWTON_JOINT_POS: dict[str, float] = {
    ".*HAA": 0.0,
    ".*F_HFE": 0.4,
    ".*H_HFE": -0.4,
    ".*F_KFE": -0.8,
    ".*H_KFE": 0.8,
    "dynaarm_shoulder_rotation": ANYMAL_HOME_ARM[0],
    "dynaarm_shoulder_flexion": ANYMAL_HOME_ARM[1],
    "dynaarm_elbow_flexion": ANYMAL_HOME_ARM[2],
    "dynaarm_forearm_rotation": ANYMAL_HOME_ARM[3],
    "dynaarm_wrist_flexion": ANYMAL_HOME_ARM[4],
    "dynaarm_wrist_rotation": ANYMAL_HOME_ARM[5],
    **ROBOTIQ_INIT_JOINT_POS,
}

# Newton binary motor targets [rad]. Followers are constrained in the USD.
ANYMAL_NEWTON_GRIPPER_OPEN = (0.0,)
ANYMAL_NEWTON_GRIPPER_CLOSE = (0.7,)

ANYMAL_ARM_BENCH_CFG = ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.replace(
        articulation_props=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.articulation_props.replace(
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ANYMAL_BASE_POS,
        joint_pos=ANYMAL_BENCH_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
)
"""Fixed-base ANYmal-D + DynaArm + 2F-140 at the Spot bench board/valve."""

# Baked kitless assembly (see scripts/generate_anymal_newton_usd.py): ANYmal-D
# + DynaArm + 2F-140 welded, no Isaac Sim spawn funcs needed at runtime.
ANYMAL_BAKED_USDA = str(Path(__file__).resolve().parent / "usd" / "anymal_d_dynaarm_robotiq.usda")

ANYMAL_ARM_NEWTON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ANYMAL_BAKED_USDA,
        activate_contact_sensors=True,
        # PhysX: disable_gravity on the root (nested-stop). Newton: per-body
        # gravcomp is applied at startup on every robot link (see events).
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        # USD already articulates from the ANYmal trunk; do not set
        # fix_root_link (PhysX-only writer).
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ANYMAL_BASE_POS,
        joint_pos=ANYMAL_NEWTON_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fixed_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            effort_limit=90.0,
            stiffness=60.0,
            damping=2.0,
            friction=0.02,
            armature=0.01,
        ),
        "dynaarm_shoulder": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_shoulder_rotation", "dynaarm_shoulder_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "dynaarm_elbow": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_elbow_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=200.0,
            damping=20.0,
        ),
        # Roll joints drive tiny reflected inertias (wrist_2 is 48 g); the
        # flat 200/20 group rang them into a ~20 rad/s limit cycle at the
        # 50 Hz control rate. Soft per-joint gains + rotor armature keep the
        # discrete PD stable (kp*dt^2/I < 1 with I >= armature).
        "dynaarm_forearm": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_forearm_rotation"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=40.0,
            damping=1.5,
            armature=0.01,
        ),
        # Wrist flexion carries the same light payload as the roll joints
        # above (wrist_2 + flange + the 0.63 kg gripper assembly) and needs the
        # same treatment. On the flat 200/20 gains it held a steady 0.29 rad
        # offset from its commanded angle with the actuator pinned at its
        # 40 N.m ceiling, while every other arm joint tracked to 0.002 rad -
        # gravity on that payload is ~1.3 N.m, so the torque was the discrete
        # PD fighting itself. That 0.29 rad is 16.6 deg of TCP orientation
        # error and ~9 cm of position error, which kept the ball-valve
        # approach from ever reaching its 0.02 m threshold: the sequence
        # stalled at the grasp and no demonstration could succeed. Raising the
        # effort ceiling instead makes it worse (the ringing gets more
        # authority); soft gains plus armature fix it.
        "dynaarm_wrist_flex": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_wrist_flexion"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=40.0,
            damping=1.5,
            armature=0.01,
        ),
        "dynaarm_wrist_rot": IdealPDActuatorCfg(
            joint_names_expr=["dynaarm_wrist_rotation"],
            effort_limit=40.0,
            velocity_limit=4.0,
            stiffness=20.0,
            damping=0.9,
            armature=0.01,
        ),
        "gripper": IdealPDActuatorCfg(
            joint_names_expr=list(NEWTON_GRIPPER_JOINT_NAMES),
            effort_limit=40.0,
            velocity_limit=1.5,
            stiffness=80.0,
            damping=4.0,
            armature=0.002,
        ),
        "gripper_linkage": IdealPDActuatorCfg(
            joint_names_expr=[
                "right_outer_knuckle_joint",
                ".*_outer_finger_joint",
                ".*_inner_finger_joint",
                ".*_inner_finger_pad_joint",
            ],
            effort_limit=40.0,
            stiffness=0.0,
            damping=0.1,
            armature=0.002,
        ),
    },
)
"""Kitless Newton ANYmal assembly (baked USD, explicit IdealPD).

Newton MJWarp ignores runtime Implicit stiffness writes, so this config uses
explicit IdealPD (Lab computes torque and forwards it as joint_f), mirroring
the Spot bench config. Gripper gains match the Isaac Sim Robotiq setup.
"""
