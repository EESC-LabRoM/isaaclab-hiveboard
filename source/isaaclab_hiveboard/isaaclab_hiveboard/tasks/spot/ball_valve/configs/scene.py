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
BALL_VALVE_URDF = f"{HIVEBOARD_SIM_DIR}/Valves/Lever Valve/Ball Valve/Ball_Valve.urdf"
HONEYCOMB_USD = f"{HIVEBOARD_SIM_DIR}/Honeycomb/Honeycomb_Panel.usd"

##
# Configuration
##


@configclass
class BallValveSceneCfg(InteractiveSceneCfg):
    """Configuration for the ball valve scene with a robot.

    This is the abstract base implementation, the exact scene is defined in the derived classes
        which need to set the robot and end-effector frames
    """

    # robot
    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Isaac Sim small warehouse (Nucleus). Shared world backdrop, not cloned
    # per env. Dropped 0.60 m so the floor matches the old ground plane and
    # the fixed-base Spot at z=0. Collision is off so shelves do not hit the
    # arm or HiveBoard valve.
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.60)),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ball_valve = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Valve",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=BALL_VALVE_URDF,
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
            pos=(1.0, 0, 0.0),
            rot=(1.0, 0, 0, 0),
            joint_pos={
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "joint_actuator": ImplicitActuatorCfg(
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

    # Visual honeycomb panel, parented under the valve body so it rides
    # along whenever the valve root pose is written. No physics/collision.
    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Valve/valvula_esfera/Honeycomb",
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

    # Frame definitions for the lever handle.
    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Valve/alavanca_pivot",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/ValveTransformers"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/alavanca_pivot",
                name="approaching",
                offset=OffsetCfg(
                    pos=(0.12, 0.06, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/alavanca_pivot",
                name="lever_pivot",
                offset=OffsetCfg(
                    pos=(0.03, 0.06, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),  # align with end-effector frame
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/alavanca_pivot",
                name="rotate_frame",
                offset=OffsetCfg(
                    pos=(0.03, -0.0, -0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),  # align with end-effector frame
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
