# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 (FR3) articulation configs for HiveBoard tasks.

The USD is the Isaac Sim asset at ``Robots/FrankaRobotics/FrankaFR3/fr3.usd``.
Link and joint names follow the official FR3 description (``fr3_*``), not the
legacy Panda (``panda_*``) names.

Reference:
    https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_robots.html
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

FRANKA_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaFR3/fr3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
            armature=1e-3,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka Research 3 with the default Franka Hand."""


FRANKA_FR3_HIGH_PD_CFG = FRANKA_FR3_CFG.copy()
FRANKA_FR3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].damping = 80.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].damping = 80.0
"""Franka Research 3 with stiffer PD gains for differential IK."""
