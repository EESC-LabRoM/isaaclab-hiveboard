# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy asset exports.

Keeping these imports lazy is essential for kitless Newton tasks: legacy robot
assets still use Isaac Sim's development-time URDF converter.
"""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[4]
ASSET_DIR = str(Path(__file__).resolve().parent)
SPOT_ASSET_DIR = str(Path(ASSET_DIR) / "spot")

_EXPORT_MODULES = {
    # Canonical end-effector profiles.
    "ANYMAL_EE": ".end_effector",
    "ANYMAL_WORKSPACE": ".end_effector",
    "FRANKA_EE": ".end_effector",
    "FRANKA_WORKSPACE": ".end_effector",
    "SPOT_EE": ".end_effector",
    "SPOT_WORKSPACE": ".end_effector",
    "EndEffectorCfg": ".end_effector",
    "WorkspaceCfg": ".end_effector",
    "as_command_offset": ".end_effector",
    "as_ik_offset": ".end_effector",
    "make_ee_frame": ".end_effector",
    # HiveBoard paths.
    "BALL_VALVE_FRICTION_RING_URDF": ".hiveboard",
    "BALL_VALVE_URDF": ".hiveboard",
    "BALL_VALVE_USD": ".hiveboard",
    "BUTTON_URDF": ".hiveboard",
    "BUTTON_USD": ".hiveboard",
    "CIRCUIT_BREAKER_URDF": ".hiveboard",
    "CIRCUIT_BREAKER_USD": ".hiveboard",
    "DRAWER_URDF": ".hiveboard",
    "DRAWER_USD": ".hiveboard",
    "HIGH_TORQUE_VALVE_URDF": ".hiveboard",
    "HIGH_TORQUE_VALVE_USD": ".hiveboard",
    "HIVEBOARD_DIR": ".hiveboard",
    "HIVEBOARD_SIM_DIR": ".hiveboard",
    "HONEYCOMB_URDF": ".hiveboard",
    "HONEYCOMB_USD": ".hiveboard",
    "KEY_URDF": ".hiveboard",
    "KEY_USD": ".hiveboard",
    "LAMP_URDF": ".hiveboard",
    "LAMP_USD": ".hiveboard",
    "LAMP_NEWTON_USD": ".hiveboard",
    "SHOCK_ABSORBER_URDF": ".hiveboard",
    "SHOCK_ABSORBER_USD": ".hiveboard",
    "SMALL_VALVE_URDF": ".hiveboard",
    "SMALL_VALVE_USD": ".hiveboard",
    # Legacy robot assets, loaded only when requested.
    "ANYMAL_D_DYNAARM_ROBOTIQ_CFG": ".anymal",
    "ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG": ".anymal",
    "ARM_PRIM": ".anymal",
    "DYNAARM_EE_LINK": ".anymal",
    "DYNAARM_JOINT_NAMES": ".anymal",
    "DYNAARM_MOUNT_POS": ".anymal",
    "DYNAARM_MOUNT_ROT": ".anymal",
    "DYNAARM_URDF": ".anymal",
    "ROBOTIQ_2F140_CFG": ".anymal",
    "ROBOTIQ_CLOSE_Q": ".anymal",
    "ROBOTIQ_DRIVE_JOINT": ".anymal",
    "ROBOTIQ_INIT_JOINT_POS": ".anymal",
    "ROBOTIQ_JOINT_GEAR": ".anymal",
    "ROBOTIQ_OPEN_Q": ".anymal",
    "robotiq_joint_targets": ".anymal",
    "FRANKA_FR3_CFG": ".franka",
    "FRANKA_FR3_HIGH_PD_CFG": ".franka",
}

__all__ = ["ASSET_DIR", "REPO_DIR", "SPOT_ASSET_DIR", *_EXPORT_MODULES]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value
