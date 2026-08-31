from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import FRANKA_FR3_HIGH_PD_CFG, FRANKA_WORKSPACE, FRANKA_EE, make_ee_frame
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import LampSceneCfg


@configclass
class FrankaLampSceneCfg(LampSceneCfg):
    """Franka Research 3 facing the shared screw-in lamp."""

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
        self.lamp.init_state.pos = FRANKA_WORKSPACE.object_pos
        self.lamp.init_state.rot = FRANKA_WORKSPACE.object_rot
