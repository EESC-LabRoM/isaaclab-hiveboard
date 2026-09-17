import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_hiveboard.assets import FRANKA_FR3_HIGH_PD_CFG, FRANKA_WORKSPACE, FRANKA_EE, make_ee_frame
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import (
    LAMP_SEATED_POSITION,
    LampSceneCfg,
)

# Distance from the lamp pivot to Franka's canonical TCP at grasp [m].
# Decrease this value to move the gripper deeper toward the socket; increase
# it to stop farther away. The shared default is 0.076 m.
FRANKA_LAMP_GRASP_OFFSET_X = 0.055


@configclass
class FrankaLampSceneCfg(LampSceneCfg):
    """Franka Research 3 facing the shared screw-in lamp."""

    # The shared Spot lamp scene currently omits its warehouse. Keep the
    # Franka variant visually consistent with the other Franka task scenes.
    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=FRANKA_WORKSPACE.warehouse_pos),
        collision_group=-1,
    )

    robot: ArticulationCfg = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "fr3_joint1": 0.0, "fr3_joint2": -0.569, "fr3_joint3": 0.0,
                "fr3_joint4": -2.810, "fr3_joint5": 0.0, "fr3_joint6": 3.037,
                "fr3_joint7": 0.741, "fr3_finger_joint.*": 0.04,
            },
        ),
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(FRANKA_EE)
    # The shared lamp scene defines Spot-only contact sensors.  Franka uses
    # different link names and this task does not consume contact readings.
    finger_contact = None
    jaw_contact = None

    def __post_init__(self):
        self.warehouse.init_state.pos = FRANKA_WORKSPACE.warehouse_pos
        self.lamp.init_state.pos = FRANKA_WORKSPACE.object_pos
        self.lamp.init_state.rot = FRANKA_WORKSPACE.object_rot
        # This task removes the bulb, so it must begin fully seated.
        self.lamp.init_state.joint_pos["PrismaticJoint"] = LAMP_SEATED_POSITION
        self.lamp.init_state.joint_pos["RevoluteJoint"] = 0.0

        # Keep the clearance pose close enough that the following engagement
        # is a short, clearly axial motion into the bulb. This is Franka-only;
        # Spot retains its larger arm/hand clearance.
        for target_frame in self.target_frame.target_frames:
            if target_frame.name in ("approaching", "unwind_frame"):
                target_frame.offset.pos = (0.105, 0.0, 0.0)
            elif target_frame.name in ("lamp_grasp", "screw_frame"):
                target_frame.offset.pos = (FRANKA_LAMP_GRASP_OFFSET_X, 0.0, 0.0)

        # The revolute joint is continuous and receives an accumulated target
        # over all sixteen turns. Bound the implicit drive so target lag or a
        # coordinate wrap cannot launch the ungrasped bulb into a fast spin.
        rotation_actuator = self.lamp.actuators["rotation"]
        rotation_actuator.velocity_limit_sim = 0.75
        rotation_actuator.effort_limit = None
        rotation_actuator.effort_limit_sim = 5.0
