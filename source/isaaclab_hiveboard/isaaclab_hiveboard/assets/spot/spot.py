# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

"""Configuration for the Boston Dynamics robot.

The following configuration parameters are available:

* :obj:`SPOT_ARM_CFG`: The Spot Arm robot with delay PD and remote PD actuators.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import (
    DelayedPDActuatorCfg,
    ImplicitActuatorCfg,
)
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.assets.spot.actuators import SpotKneeActuatorCfg
from isaaclab_hiveboard.assets.spot.constants import (
    ARM_ARMATURE,
    ARM_DAMPING,
    ARM_EFFORT_LIMIT,
    ARM_STIFFNESS,
    HIP_DAMPING,
    HIP_EFFORT_LIMIT,
    HIP_STIFFNESS,
    JOINT_PARAMETER_LOOKUP_TABLE,
    KNEE_DAMPING,
    KNEE_STIFFNESS,
    SPOT_DEFAULT_JOINT_POS,
    SPOT_DEFAULT_POS,
)

##
# Configuration
##

SPOT_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=False,
        link_density=1.0e-8,
        asset_path=f"{ASSET_DIR}/spot/spot_with_arm.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "robot")],
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=SPOT_DEFAULT_POS,
        joint_pos=SPOT_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=HIP_EFFORT_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_knee": SpotKneeActuatorCfg(
            joint_names_expr=[".*_kn"],
            joint_parameter_lookup=JOINT_PARAMETER_LOOKUP_TABLE,
            effort_limit=None,  # torque limits are handled based experimental data
            stiffness=KNEE_STIFFNESS,
            damping=KNEE_DAMPING,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
            enable_torque_speed_limit=True,
        ),
        "spot_arm_sh0": ImplicitActuatorCfg(
            joint_names_expr=["arm_sh0"],
            effort_limit=ARM_EFFORT_LIMIT[0],
            stiffness=ARM_STIFFNESS[0],
            damping=ARM_DAMPING[0],
            armature=ARM_ARMATURE[0],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_sh1": ImplicitActuatorCfg(
            joint_names_expr=["arm_sh1"],
            effort_limit=ARM_EFFORT_LIMIT[1],
            stiffness=ARM_STIFFNESS[1],
            damping=ARM_DAMPING[1],
            armature=ARM_ARMATURE[1],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el0": ImplicitActuatorCfg(
            joint_names_expr=["arm_el0"],
            effort_limit=ARM_EFFORT_LIMIT[2],
            stiffness=ARM_STIFFNESS[2],
            damping=ARM_DAMPING[2],
            armature=ARM_ARMATURE[2],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_el1": ImplicitActuatorCfg(
            joint_names_expr=["arm_el1"],
            effort_limit=ARM_EFFORT_LIMIT[3],
            stiffness=ARM_STIFFNESS[3],
            damping=ARM_DAMPING[3],
            armature=ARM_ARMATURE[3],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr0": ImplicitActuatorCfg(
            joint_names_expr=["arm_wr0"],
            effort_limit=ARM_EFFORT_LIMIT[4],
            stiffness=ARM_STIFFNESS[4],
            damping=ARM_DAMPING[4],
            armature=ARM_ARMATURE[4],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_wr1": ImplicitActuatorCfg(
            joint_names_expr=["arm_wr1"],
            effort_limit=ARM_EFFORT_LIMIT[5],
            stiffness=ARM_STIFFNESS[5],
            damping=ARM_DAMPING[5],
            armature=ARM_ARMATURE[5],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
        "spot_arm_f1x": ImplicitActuatorCfg(
            joint_names_expr=["arm_f1x"],
            effort_limit=ARM_EFFORT_LIMIT[6],
            stiffness=ARM_STIFFNESS[6],
            damping=ARM_DAMPING[6],
            armature=ARM_ARMATURE[6],
            # min_delay=0,  # physics time steps (min: 5.0*1=5.0ms)
            # max_delay=3,  # physics time steps (max: 5.0*2=10.0ms)
        ),
    },
)
