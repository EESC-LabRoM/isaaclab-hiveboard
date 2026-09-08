from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import SPOT_EE, make_ee_frame
from isaaclab_hiveboard.assets.spot.spot import (
    SPOT_ARM_NEWTON_CFG,
    SPOT_ARM_UUC_BODY_PRIM,
    SPOT_ARM_UUC_SOURCE_PRIM,
)
from isaaclab_hiveboard.tasks.scenes.lever_valve import (
    LeverValveSceneCfg as LeverValveSceneBase,
)


@configclass
class BallValveSceneCfg(LeverValveSceneBase):
    """Spot + shared HiveBoard lever (ball) valve scene."""

    robot: ArticulationCfg = SPOT_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_EE)

    def __post_init__(self):
        # UUC nests links under Geometry/body/...; body names stay arm_link_wr1.
        self.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_SOURCE_PRIM
        wr1 = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_BODY_PRIM
        for frame in self.ee_frame.target_frames:
            if frame.name == "ee_tcp":
                frame.prim_path = wr1
