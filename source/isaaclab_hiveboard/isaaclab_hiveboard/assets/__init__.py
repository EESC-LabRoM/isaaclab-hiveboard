# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

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
