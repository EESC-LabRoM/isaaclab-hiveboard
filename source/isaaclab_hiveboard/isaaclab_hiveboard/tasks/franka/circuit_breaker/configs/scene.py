from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

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
CIRCUIT_BREAKER_URDF = (
    f"{HIVEBOARD_SIM_DIR}/Circuit Breaker/Circuit_Breaker_Assembly.urdf"
)
HONEYCOMB_USD = f"{HIVEBOARD_SIM_DIR}/Honeycomb/Honeycomb_Panel.usd"


@configclass
class FrankaCircuitBreakerSceneCfg(InteractiveSceneCfg):
    """Franka Emika Panda + HiveBoard Circuit Breaker in the Isaac warehouse."""

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    # Warehouse backdrop
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        collision_group=-1,
    )

    # Light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # Circuit Breaker asset mounted at reachable workspace height
    circuit_breaker = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=CIRCUIT_BREAKER_URDF,
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
            semantic_tags=[("class", "circuit_breaker")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.40),
            rot=(
                0.0,
                0.0,
                0.0,
                1.0,
            ),  # 180 deg about Z so front faces Franka at (0, 0, 0)
            joint_pos={
                "RevoluteJoint": 0.5235988,  # Starts in UP position (+30 deg)
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
        },
    )

    # Honeycomb panel parented under Circuit Breaker base link
    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker/World/Honeycomb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=HONEYCOMB_USD,
            scale=(0.001, 0.001, 0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            semantic_tags=[("class", "honeycomb")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.04, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        collision_group=-1,
    )

    # Frame definitions for the lever handle
    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/CircuitBreaker/World",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/CircuitBreakerTransformers"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/World",
                name="approaching",
                offset=OffsetCfg(
                    pos=(0.10, 0.02, 0.0),
                    # A source-frame pitch of -110 deg keeps the TCP strongly
                    # panel-facing (score 0.94) without the Franka's singular
                    # centerline pose at exactly -90 deg.  The jaw Y axis stays
                    # aligned with breaker Y.
                    rot=(0.5735764, 0.0, -0.8191520, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/World",
                name="lever_pivot_below",
                offset=OffsetCfg(
                    pos=(0.05, 0.0, -0.07),
                    rot=(0.5735764, 0.0, -0.8191520, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/CircuitBreaker/World",
                name="lever_pivot_above",
                offset=OffsetCfg(
                    pos=(0.03, 0.0, 0.07),
                    rot=(0.5735764, 0.0, -0.8191520, 0.0),
                ),
            ),
        ],
    )

    # Franka End-Effector Frame Transformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/EndEffectorFrameTransformer"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )
