# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

from .anymal import (
    ANYMAL_D_DYNAARM_ROBOTIQ_CFG,
    ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG,
    ARM_PRIM,
    DYNAARM_EE_LINK,
    DYNAARM_JOINT_NAMES,
    DYNAARM_MOUNT_POS,
    DYNAARM_MOUNT_ROT,
    DYNAARM_URDF,
    ROBOTIQ_2F140_CFG,
    ROBOTIQ_CLOSE_Q,
    ROBOTIQ_DRIVE_JOINT,
    ROBOTIQ_INIT_JOINT_POS,
    ROBOTIQ_JOINT_GEAR,
    ROBOTIQ_OPEN_Q,
    robotiq_joint_targets,
)
from .end_effector import (
    ANYMAL_EE,
    ANYMAL_WORKSPACE,
    FRANKA_EE,
    FRANKA_WORKSPACE,
    SPOT_EE,
    SPOT_WORKSPACE,
    EndEffectorCfg,
    WorkspaceCfg,
    as_command_offset,
    as_ik_offset,
    make_ee_frame,
)
from .franka import FRANKA_FR3_CFG, FRANKA_FR3_HIGH_PD_CFG
from .hiveboard import (
    ASSET_DIR,
    BALL_VALVE_FRICTION_RING_URDF,
    BALL_VALVE_URDF,
    BALL_VALVE_USD,
    BUTTON_URDF,
    BUTTON_USD,
    CIRCUIT_BREAKER_URDF,
    CIRCUIT_BREAKER_USD,
    DRAWER_URDF,
    DRAWER_USD,
    HIGH_TORQUE_VALVE_URDF,
    HIGH_TORQUE_VALVE_USD,
    HIVEBOARD_DIR,
    HIVEBOARD_SIM_DIR,
    HONEYCOMB_URDF,
    HONEYCOMB_USD,
    KEY_URDF,
    KEY_USD,
    LAMP_URDF,
    LAMP_USD,
    SHOCK_ABSORBER_URDF,
    SHOCK_ABSORBER_USD,
    SMALL_VALVE_URDF,
    SMALL_VALVE_USD,
)

REPO_DIR = Path(__file__).resolve().parents[4]
SPOT_ASSET_DIR = os.path.join(ASSET_DIR, "spot")
