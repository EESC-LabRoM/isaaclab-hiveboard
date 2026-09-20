from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, FRANKA_WORKSPACE, make_ee_frame
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LampSceneCfg


@configclass
class FrankaLampSceneCfg(LampSceneCfg):
    """Franka Research 3 facing the shared screw-in lamp."""

    robot: ArticulationCfg = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
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
    ee_frame: FrameTransformerCfg = make_ee_frame(FRANKA_EE)
    # The shared lamp scene defines Spot-only contact sensors.  Franka uses
    # different link names and this task does not consume contact readings.
    finger_contact = None
    jaw_contact = None

    def __post_init__(self):
        self.lamp.init_state.pos = FRANKA_WORKSPACE.object_pos
        # Newton applies these drives as explicit torques. Armature keeps
        # the stiff TCP controller stable at the shared 1 kHz physics rate.
        self.robot.actuators = {
            "fr3_shoulder": IdealPDActuatorCfg(
                joint_names_expr=["fr3_joint[1-4]"],
                effort_limit=87.0,
                stiffness=400.0,
                damping=80.0,
                armature=0.1,
            ),
            "fr3_forearm": IdealPDActuatorCfg(
                joint_names_expr=["fr3_joint[5-7]"],
                effort_limit=12.0,
                stiffness=400.0,
                damping=80.0,
                armature=0.05,
            ),
            "fr3_hand": IdealPDActuatorCfg(
                joint_names_expr=["fr3_finger_joint.*"],
                effort_limit=20.0,
                stiffness=200.0,
                damping=20.0,
                armature=0.01,
            ),
        }
