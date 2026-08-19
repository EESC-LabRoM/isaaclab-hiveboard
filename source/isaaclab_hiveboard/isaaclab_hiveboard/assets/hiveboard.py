# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset paths and resolution helpers for the HiveBoard benchmark."""

import os
from pathlib import Path

# Package assets directory
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

def _find_hiveboard_dir() -> str:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = current / "dependencies" / "HiveBoard"
        if candidate.exists() and (candidate / "Simulation").exists():
            return str(candidate)
        current = current.parent
    # Fallback to current directory
    return str(Path(__file__).resolve().parents[4] / "dependencies" / "HiveBoard")


HIVEBOARD_DIR = os.getenv("HIVEBOARD_DIR", _find_hiveboard_dir())
HIVEBOARD_SIM_DIR = os.path.join(HIVEBOARD_DIR, "Simulation")

# HiveBoard Panel
HONEYCOMB_USD = os.path.join(HIVEBOARD_SIM_DIR, "Honeycomb", "Honeycomb_Panel.usd")
HONEYCOMB_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Honeycomb", "Honeycomb_Panel.urdf")

# HiveBoard Interactive Objects - Valves
BALL_VALVE_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Valves", "Lever Valve", "Ball Valve", "Ball_Valve.urdf")
BALL_VALVE_USD = os.path.join(HIVEBOARD_SIM_DIR, "Valves", "Lever Valve", "Ball Valve", "Ball_Valve.usd")
BALL_VALVE_FRICTION_RING_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Lever Valve", "Ball Valve", "Ball_Valve_Friction_Ring_Set.urdf"
)
HIGH_TORQUE_VALVE_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "High Torque Valve", "High_Torque_Valve.urdf"
)
HIGH_TORQUE_VALVE_USD = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "High Torque Valve", "High_Torque_Valve.usd"
)
SMALL_VALVE_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "Small Valve", "Small_Valve.urdf"
)
SMALL_VALVE_USD = os.path.join(
    HIVEBOARD_SIM_DIR, "Valves", "Gate Valve", "Small Valve", "Small_Valve.usd"
)

# HiveBoard Interactive Objects - Electrical & Mechanical
CIRCUIT_BREAKER_URDF = os.path.join(
    HIVEBOARD_SIM_DIR, "Circuit Breaker", "Circuit_Breaker_Assembly.urdf"
)
CIRCUIT_BREAKER_USD = os.path.join(
    HIVEBOARD_SIM_DIR, "Circuit Breaker", "Circuit_Breaker_Assembly.usd"
)
BUTTON_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Button", "Button_Assembly.urdf")
BUTTON_USD = os.path.join(HIVEBOARD_SIM_DIR, "Button", "Button_Assembly.usd")
DRAWER_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Drawer", "Drawer_Assembly.urdf")
DRAWER_USD = os.path.join(HIVEBOARD_SIM_DIR, "Drawer", "Drawer_Assembly.usd")
KEY_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Key", "Key_assembly.urdf")
KEY_USD = os.path.join(HIVEBOARD_SIM_DIR, "Key", "Key_assembly.usd")
LAMP_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Lamp", "Lamp_Assembly.urdf")
LAMP_USD = os.path.join(HIVEBOARD_SIM_DIR, "Lamp", "Lamp_Assembly.usd")
SHOCK_ABSORBER_URDF = os.path.join(HIVEBOARD_SIM_DIR, "Shock Absorber", "Shock_Absorber_Assembly.urdf")
SHOCK_ABSORBER_USD = os.path.join(HIVEBOARD_SIM_DIR, "Shock Absorber", "Shock_Absorber_Assembly.usd")
