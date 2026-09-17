import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, FRANKA_WORKSPACE, make_ee_frame
from isaaclab_hiveboard.assets.hiveboard import HONEYCOMB_URDF
from isaaclab_hiveboard.tasks.spot.small_valve.configs.scene import SmallValveSceneCfg

from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
)

HALF_SQRT = math.sqrt(0.5)

# Target-pose tuning. These offsets are expressed in the fixed valve-housing
# frame. Housing -Y points vertically upward after the scene rotation, so the
# magnitude of Y is the TCP height above the valve revolute origin.
APPROACH_OFFSET = (0.0, -0.10, 0.0)
GRASP_OFFSET = (0.0, -0.028, 0.0)
GRASP_ORIENTATION = (0.5, 0.5, 0.5, 0.5)


@configclass
class FrankaGateValveSmallSceneCfg(SmallValveSceneCfg):
    """Floor-mounted panel with an upward-facing valve in its central cell."""

    robot = FRANKA_FR3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = make_ee_frame(FRANKA_EE)
    # The panel URDF is in metres, with its normal along +X. Rotate that
    # normal upward; its mounting face sits 20 mm above the floor.
    honeycomb = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Honeycomb",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=HONEYCOMB_URDF,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(HALF_SQRT, 0.0, -HALF_SQRT, 0.0),
        ),
    )
    # Use housing-fixed poses so releasing a turned stem does not rotate the
    # next grasp orientation. Root -Y is the stem axis and becomes world +Z.
    target_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Valve/valvula_gaveta",
        debug_vis=False,
        # Match the target axes to the EE axes produced by make_ee_frame().
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(
            prim_path="/Visuals/FrankaGateValveSmallTargetFrames"
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/valvula_gaveta",
                name="approaching",
                offset=OffsetCfg(pos=APPROACH_OFFSET, rot=GRASP_ORIENTATION),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/valvula_gaveta",
                name="nut_grasp",
                # Mesh_22's wheel occupies stem Z=0.021..0.038 m after
                # applying its URDF visual/collision origin offset.
                offset=OffsetCfg(pos=GRASP_OFFSET, rot=GRASP_ORIENTATION),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/valvula_gaveta",
                name="rotate_frame",
                offset=OffsetCfg(pos=GRASP_OFFSET),
            ),
        ],
    )

    def __post_init__(self):
        self.warehouse.init_state.pos = FRANKA_WORKSPACE.warehouse_pos
        # Housing mounting face is 75 mm behind the revolute origin.
        self.small_valve.init_state.pos = (0.55, 0.0, 0.095)
        self.small_valve.init_state.rot = (HALF_SQRT, -HALF_SQRT, 0.0, 0.0)
        self.small_valve.init_state.joint_pos = {"PrismaticJoint": 0.0, "RevoluteJoint": 0.0}
        # The 5 g stem otherwise coasts after release under the shared asset's
        # zero damping. Keep it passive, with drag and bounded free-spin speed.
        self.small_valve.spawn.rigid_props.angular_damping = 5.0
        self.small_valve.spawn.rigid_props.max_angular_velocity = 1.0
        self.small_valve.spawn.articulation_props.solver_position_iteration_count = 16
        self.small_valve.spawn.articulation_props.solver_velocity_iteration_count = 4
