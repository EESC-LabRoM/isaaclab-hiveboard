from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

from isaaclab_hiveboard.assets.spot.spot import SPOT_ARM_CFG

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_hiveboard.assets import HIVEBOARD_DIR

HIVEBOARD_SIM_DIR = f"{HIVEBOARD_DIR}/Simulation"
HIGH_TORQUE_VALVE_URDF = f"{HIVEBOARD_SIM_DIR}/Valves/Gate Valve/High Torque Valve/High_Torque_Valve.urdf"
HONEYCOMB_USD = f"{HIVEBOARD_SIM_DIR}/Honeycomb/Honeycomb_Panel.usd"

# +90 deg about Y: used on command frames so RotateFrame's -X lines up with
# the nut spin axis (CAD +Z).
VALVE_Y90_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
# -90 deg about Y: CAD +Z (handwheel) points toward Spot at the origin.
VALVE_SPAWN_POS = (1.0, 0.0, 0.0)
VALVE_SPAWN_QUAT = (0.70710678, 0.0, -0.70710678, 0.0)
# Inverse of the valve spawn rotation so the hive stays wall-aligned.
HIVE_SPAWN_INV_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)

# Flange (arm_link_wr1) → TCP. Shared by ee_frame, pose commands, and reset IK.
EE_TCP_OFFSET = OffsetCfg(pos=(0.21, 0.0, -0.03))
# Nut-frame targets for target_frame. Reset IK looks up ``approaching`` by
# name so the arm spawns there and the sequence can skip an arrival command.
VALVE_APPROACHING_OFFSET = OffsetCfg(pos=(-0.04, 0.0, 0.25), rot=VALVE_Y90_QUAT)
VALVE_NUT_GRASP_OFFSET = OffsetCfg(pos=(-0.04, 0.0, 0.14), rot=VALVE_Y90_QUAT)
VALVE_ROTATE_OFFSET = OffsetCfg(pos=(-0.0, 0.0, 0.0), rot=VALVE_Y90_QUAT)


@configclass
class HighTorqueValveSceneCfg(InteractiveSceneCfg):
    """Spot + HiveBoard high-torque (gate) valve in the Isaac warehouse."""

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.60)),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    high_torque_valve = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=HIGH_TORQUE_VALVE_URDF,
            activate_contact_sensors=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None),  # type: ignore
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            semantic_tags=[("class", "valve")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=VALVE_SPAWN_POS,
            # CAD +Z is the handwheel axis; -90° about Y aims it at the robot.
            rot=VALVE_SPAWN_QUAT,
            joint_pos={
                "PrismaticJoint": 0.0,
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "revolute_actuator": ImplicitActuatorCfg(
                damping=0.0,
                friction=0.02,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=2.0,
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.0,
            ),
            "prismatic_fixed": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                stiffness=1e5,
                damping=1e3,
                effort_limit=1000.0,
            ),
        },
    )

    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Valve/World/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # Behind the valve in the rotated World frame. Local +90° Y undoes
            # the valve spawn so the panel stays upright as before.
            pos=(0.0, 0.0, 0.0),
            rot=HIVE_SPAWN_INV_QUAT,
        ),
        collision_group=-1,
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Valve/nut",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ValveTransformers"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/nut",
                name="approaching",
                offset=VALVE_APPROACHING_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/nut",
                name="nut_grasp",
                offset=VALVE_NUT_GRASP_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/nut",
                name="rotate_frame",
                offset=VALVE_NUT_GRASP_OFFSET,
            ),
        ],
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="tool_rightfinger",
                offset=EE_TCP_OFFSET,
            ),
        ],
    )
