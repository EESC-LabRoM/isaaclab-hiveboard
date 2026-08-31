from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

from isaaclab_hiveboard.assets import (
    HONEYCOMB_USD,
    LAMP_URDF,
    SPOT_EE,
    make_ee_frame,
)
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

# Turn the lamp's local +X axis toward Spot. The socket remains behind it.
LAMP_SPAWN_POS = (1.0, 0.0, 0.0)
LAMP_SPAWN_QUAT = (0.0, 0.0, 0.0, 1.0)
SPOT_FORWARD_OFFSET = 0.030
LAMP_ROOT_POS_IN_ROBOT_BASE = (
    LAMP_SPAWN_POS[0] - SPOT_FORWARD_OFFSET,
    LAMP_SPAWN_POS[1],
    LAMP_SPAWN_POS[2],
)
LAMP_UNSCREWED_POSITION = 0.024
LAMP_SEATED_POSITION = 0.0

# All offsets are expressed in the lamp's local frame. The lamp bulb extends
# along +X and the prismatic joint translates it by another 24 mm when open.
# The local 180-degree rotation cancels the articulation spawn rotation for
# orientation only, making canonical TCP +X point from Spot toward the lamp.
LAMP_TARGET_QUAT = LAMP_SPAWN_QUAT
# Offsets from ``lamp_pivot``. They follow the bulb inward as the screw
# coupling advances it; the grasp stays near the accessible front surface.
LAMP_APPROACH_OFFSET = OffsetCfg(pos=(0.126, 0.0, 0.0), rot=LAMP_TARGET_QUAT)
# Forty millimetres deeper than the original surface grasp. The TCP is shifted
# 30 mm below the bulb axis so the lower part of Spot's open hand can pass the
# bulb's widest section before the fingers close.
LAMP_GRASP_OFFSET = OffsetCfg(pos=(0.076, 0.0, -0.00), rot=LAMP_TARGET_QUAT)
LAMP_ROTATION_OFFSET = OffsetCfg(pos=(0.076, 0.0, 0.0), rot=LAMP_TARGET_QUAT)
# Pull straight off the bulb while still loosely closed, then open and unwind
# at this clearance pose.  Opening at the deep grasp pose expands the fingers
# through the bulb collision mesh and produces a large contact impulse.
LAMP_RELEASE_OFFSET = OffsetCfg(pos=(0.126, 0.0, -0.030), rot=LAMP_TARGET_QUAT)
LAMP_RESET_APPROACH_OFFSET = OffsetCfg(pos=(0.15, 0.0, 0.0), rot=LAMP_TARGET_QUAT)


@configclass
class LampSceneCfg(InteractiveSceneCfg):
    """Fixed-base Spot facing a screw-in lamp on the HiveBoard panel."""

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # warehouse = AssetBaseCfg(
    #     prim_path="/World/Warehouse",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.60)),
    #     collision_group=-1,
    # )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    lamp = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Lamp",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            link_density=1.0e-8,
            asset_path=LAMP_URDF,
            activate_contact_sensors=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None),  # type: ignore
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=False,
                linear_damping=2.0,
                angular_damping=1.0,
                max_linear_velocity=10.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=0.20,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            semantic_tags=[("class", "lamp")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=LAMP_SPAWN_POS,
            rot=LAMP_SPAWN_QUAT,
            joint_pos={
                "PrismaticJoint": LAMP_UNSCREWED_POSITION,
                "RevoluteJoint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "rotation": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                # Compliant PD so ScrewFrame can turn the bulb with the gripper
                # instead of sweeping the fingers through a stationary mesh.
                stiffness=80.0,
                damping=8.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=100.0,
            ),
            "insertion": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                # The environment updates this drive target from measured
                # revolute travel using the lamp's 6 mm/revolution pitch. The
                # USD values (50 kN/m, 300 N-s/m) are too stiff for the URDF's
                # light bulb link at this timestep, so use a stable equivalent.
                stiffness=5000.0,
                damping=50.0,
                friction=0.0,
                dynamic_friction=0.0,
                viscous_friction=0.0,
                effort_limit=100.0,
            ),
        },
    )

    # honeycomb = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Lamp/World/Honeycomb",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=HONEYCOMB_USD,
    #         scale=(0.001, 0.001, 0.001),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         semantic_tags=[("class", "honeycomb")],
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         rot=LAMP_SPAWN_QUAT,
    #     ),
    #     collision_group=-1,
    # )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LampTransformers"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
                name="approaching",
                offset=LAMP_APPROACH_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
                name="lamp_grasp",
                offset=LAMP_GRASP_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
                name="screw_frame",
                offset=LAMP_ROTATION_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
                name="release_frame",
                offset=LAMP_RELEASE_OFFSET,
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Lamp/lamp_pivot",
                name="unwind_frame",
                offset=LAMP_APPROACH_OFFSET,
            ),
        ],
    )

    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_EE)

    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link_fngr",
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/lamp_pivot"],
        visualizer_cfg=CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/LampFingerContact"),
    )

    jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link_jaw",
        update_period=0.0,
        debug_vis=False,
        track_pose=True,
        force_threshold=0.1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Lamp/lamp_pivot"],
        visualizer_cfg=CONTACT_SENSOR_MARKER_CFG.replace(prim_path="/Visuals/LampJawContact"),
    )
