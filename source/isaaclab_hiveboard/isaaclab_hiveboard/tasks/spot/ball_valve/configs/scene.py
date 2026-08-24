from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import SPOT_EE, make_ee_frame
from isaaclab_hiveboard.assets.spot.spot import SPOT_ARM_CFG
from isaaclab_hiveboard.tasks.scenes.lever_valve import (
    LeverValveSceneCfg as LeverValveSceneBase,
)


@configclass
class BallValveSceneCfg(LeverValveSceneBase):
    """Spot + shared HiveBoard lever (ball) valve scene."""

    robot: ArticulationCfg = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_EE)
