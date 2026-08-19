import os

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from isaaclab_hiveboard.assets import ASSET_DIR
from isaaclab_hiveboard.mdp.actions.rmp_flow_action_2 import (
    RMPFlowActionCfg2,
)
from isaaclab_hiveboard.mdp.actions.rmp_flow_controller_2 import (
    RmpFlowControllerCfg2,
)

SPOT_ARM_RMPFLOW_CFG = RmpFlowControllerCfg2(
    config_file=os.path.join(
        ASSET_DIR,
        "spot",
        "rmpflow",
        "spot_rmpflow_config.yaml",
    ),
    urdf_file=f"{ASSET_DIR}/spot/spot_with_arm.urdf",
    collision_file=os.path.join(
        ASSET_DIR,
        "spot",
        "rmpflow",
        "spot_lula_config.yaml",
    ),
    frame_name="arm_link_jaw",
    evaluations_per_frame=5,
    ignore_robot_state_updates=True,
)


@configclass
class RMPActionsCfg:
    """Action specifications for the MDP."""

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.57},
        close_command_expr={"arm_f1x": 0.0},
    )

    arm_action: RMPFlowActionCfg2 = RMPFlowActionCfg2(
        asset_name="robot",
        body_name="arm_link_jaw",
        joint_names=[
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
        ],
        scale=1.0,
        controller=SPOT_ARM_RMPFLOW_CFG,
        articulation_prim_expr="/World/envs/env_.*/Robot",
        use_relative_mode=False,
        debug_vis=True,
        body_offset=RMPFlowActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


@configclass
class SpotIKRelativeActionCfg:
    """action config for Spot arm in relative mode."""

    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
        ],
        body_name="arm_link_jaw",
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        ),
        scale=1,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0)
        ),
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_f1x"],
        open_command_expr={"arm_f1x": -1.57},
        close_command_expr={"arm_f1x": 0.0},
    )
