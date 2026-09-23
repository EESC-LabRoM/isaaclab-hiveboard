import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

from isaaclab_hiveboard.assets import (
    BUTTON_URDF,
    FRANKA_EE,
    FRANKA_FR3_HIGH_PD_CFG,
    FRANKA_WORKSPACE,
    HONEYCOMB_URDF,
    make_ee_frame,
)

HALF_SQRT = math.sqrt(0.5)
FLOOR_MOUNT_ROT = (HALF_SQRT, 0.0, -HALF_SQRT, 0.0)
# The canonical Franka TCP approaches along +X. Combined with FLOOR_MOUNT_ROT,
# this 180-degree local-Y rotation makes TCP +X point vertically downward,
# matching the proven floor-mounted small-valve grasp orientation.
TCP_DOWN_ROT = (0.0, 0.0, 1.0, 0.0)

# Offsets are in the button housing frame. Housing +X is world +Z after the
# floor-mount rotation, so increasing X raises the TCP above the attachment.
# The lid hinge is authored in the button housing frame.  At the closed joint
# limit its opposite/free edge is near (x=0.017, y=-0.028).  Approach that edge
# vertically with open jaws, then descend until the edge is between the fingers.
COVER_HINGE_OFFSET = (-0.0034325, 0.0304632, 0.0)
# Descend over the free edge with the jaw axis across the cover's front/back
# thickness.  In the housing frame TCP +X is down (-X), TCP +Y is depth (+Z),
# and TCP +Z points toward the hinge (+Y).  The 80 mm open jaws therefore
# straddle the 56 mm cover before closing on its two broad side faces.  A
# 15-degree rotation about TCP -Y makes the grasp oblique at the free border;
# this raises the rear of the hand during the arc and keeps it clear of the
# floor.  ``COVER_GRASP_ROT`` is the base orientation post-multiplied by that
# local-frame tilt.
COVER_GRASP_OFFSET = (0.006, -0.025, 0.0)
_cover_tilt_half = 0.5 * math.radians(60.0)
_cover_tilt_sin = math.sin(_cover_tilt_half)
_cover_tilt_cos = math.cos(_cover_tilt_half)
COVER_GRASP_ROT = (
    HALF_SQRT * _cover_tilt_sin,
    HALF_SQRT * _cover_tilt_sin,
    HALF_SQRT * _cover_tilt_cos,
    HALF_SQRT * _cover_tilt_cos,
)
COVER_CLEARANCE_OFFSET = (0.080, -0.025, 0.0)
POST_COVER_CLEARANCE_OFFSET = (0.140, 0.110, 0.0)
BUTTON_CLEARANCE_OFFSET = (0.075, 0.0, 0.0)
BUTTON_PRESSED_OFFSET = (0.003, 0.0, 0.0)

# Stay 0.0001 rad inside the authored URDF limit. The URDF-to-USD conversion
# quantizes the bound, so using either ``-math.pi / 2`` or the exact source
# decimal can still compare microscopically below the runtime lower limit.
COVER_CLOSED_POSITION = -1.5707
COVER_OPEN_POSITION = math.radians(40.0)
MINIMUM_COVER_SUCCESS_POSITION = COVER_CLOSED_POSITION + math.pi / 2.0
BUTTON_RELEASED_POSITION = 0.0
BUTTON_PRESSED_POSITION = -0.010


@configclass
class FrankaButtonSceneCfg(InteractiveSceneCfg):
    """Floor-mounted HiveBoard with a covered push button in its center cell."""

    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRANKA_WORKSPACE.warehouse_pos),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
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
    )
    ee_frame = make_ee_frame(FRANKA_EE)

    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=HONEYCOMB_URDF,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                ),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=FLOOR_MOUNT_ROT
        ),
    )
    button = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Button",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=BUTTON_URDF,
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            activate_contact_sensors=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None),  # type: ignore
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            semantic_tags=[("class", "button")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.040),
            rot=FLOOR_MOUNT_ROT,
            joint_pos={
                "PrismaticJoint": BUTTON_RELEASED_POSITION,
                "RevoluteJoint": COVER_CLOSED_POSITION,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "button": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                stiffness=35.0,
                damping=2.0,
                friction=0.0,
                effort_limit_sim=20.0,
            ),
            "cover": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.0,
                damping=0.05,
                friction=0.02,
                effort_limit_sim=2.0,
            ),
        },
    )

    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Button/World",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/FrankaButtonTargetFrames"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="cover_clearance",
                offset=OffsetCfg(pos=COVER_CLEARANCE_OFFSET, rot=COVER_GRASP_ROT),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="cover_grasp",
                offset=OffsetCfg(pos=COVER_GRASP_OFFSET, rot=COVER_GRASP_ROT),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="cover_hinge",
                offset=OffsetCfg(pos=COVER_HINGE_OFFSET),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="post_cover_clearance",
                offset=OffsetCfg(pos=POST_COVER_CLEARANCE_OFFSET, rot=TCP_DOWN_ROT),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="button_clearance",
                offset=OffsetCfg(pos=BUTTON_CLEARANCE_OFFSET, rot=TCP_DOWN_ROT),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Button/World",
                name="button_pressed",
                offset=OffsetCfg(pos=BUTTON_PRESSED_OFFSET, rot=TCP_DOWN_ROT),
            ),
        ],
    )
