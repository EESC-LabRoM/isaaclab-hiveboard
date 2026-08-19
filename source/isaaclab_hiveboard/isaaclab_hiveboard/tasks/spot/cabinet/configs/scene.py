from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_hiveboard.assets.spot.spot import SPOT_ARM_CFG

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

##
# Configuration
##


@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robot
    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector
    # Listens to the required transforms
    # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
    # the other frames are the fingers
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/EndEffectorFrameTransformer"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.21, 0.0, -0.01),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_fngr",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.095456954, 0.0, -0.04),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link_wr1",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.21, 0.0, -0.04),
                ),
            ),
        ],
    )

    cube: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.1, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.0, 0.8)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.10, -0.25), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # Frame definitions for the cabinet.
    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/CabinetFrameTransformer"
        ),
        target_frames=[
            # This frame should be the desired position of the end effector
            FrameTransformerCfg.FrameCfg(
                # prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                prim_path="{ENV_REGEX_NS}/Cube",  # /drawer_handle_top",
                name="drawer_handle_top_2",
                offset=OffsetCfg(
                    pos=(-0, 0.25, 0.10),
                    rot=(1.0, 0.0, 0.0, 0.0),  # align with end-effector frame
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                # prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                prim_path="{ENV_REGEX_NS}/Cube",  # /drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(-0, -0.45, 0.10),
                    rot=(1.0, 0.0, 0.0, 0.0),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.60)),
        spawn=sim_utils.GroundPlaneCfg(
            semantic_tags=[("class", "ground")],
        ),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
