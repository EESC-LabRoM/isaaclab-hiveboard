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
SMALL_VALVE_URDF = f"{HIVEBOARD_SIM_DIR}/Valves/Gate Valve/Small Valve/Small_Valve.urdf"
HONEYCOMB_USD = f"{HIVEBOARD_SIM_DIR}/Honeycomb/Honeycomb_Panel.usd"

VALVE_Y90_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
HIVE_Y90_INV_QUAT = (
    0.65328148,
    0.27059805,
    -0.27059805,
    -0.65328148,
)  # (0.70710678, 0.0, 0.0, -0.70710678)


@configclass
class SmallValveSceneCfg(InteractiveSceneCfg):
    """Spot + HiveBoard small gate valve in the Isaac warehouse."""

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

    small_valve = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=SMALL_VALVE_URDF,
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
            pos=(0.1, 0, 0.0),
            rot=VALVE_Y90_QUAT,
            joint_pos={
                "PrismaticJoint": 0.0,
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "revolute_actuator": ImplicitActuatorCfg(
                damping=0.0,
                friction=0.0002,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=0.0,
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.00000,
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
        prim_path="{ENV_REGEX_NS}/Valve/valvula_gaveta/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.075, 0.0),
            rot=HIVE_Y90_INV_QUAT,
        ),
        collision_group=-1,
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Valve/eixo_trans",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/ValveTransformers"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/eixo_trans",
                name="approaching",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.20),
                    rot=VALVE_Y90_QUAT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/eixo_trans",
                name="nut_grasp",
                offset=OffsetCfg(
                    pos=(-0.02, 0.0, 0.05),
                    rot=VALVE_Y90_QUAT,
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/eixo_trans",
                name="rotate_frame",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.05),
                    rot=VALVE_Y90_QUAT,
                ),
            ),
        ],
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/EndEffectorFrameTransformer"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.21, 0.0, -0.03),
                ),
            ),
        ],
    )
