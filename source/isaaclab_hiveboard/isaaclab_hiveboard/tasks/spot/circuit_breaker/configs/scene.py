from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import SPOT_EE, make_ee_frame
from isaaclab_hiveboard.assets.spot.spot import (
    SPOT_ARM_NEWTON_CFG,
    SPOT_ARM_UUC_BODY_PRIM,
    SPOT_ARM_UUC_FNGR_PRIM,
    SPOT_ARM_UUC_JAW_PRIM,
    SPOT_ARM_UUC_SOURCE_PRIM,
)
from isaaclab_hiveboard.tasks.scenes.circuit_breaker import (
    CircuitBreakerSceneCfg as CircuitBreakerSceneBase,
)

_BREAKER_CONTACT_FILTER = ["{ENV_REGEX_NS}/CircuitBreaker/.*"]


@configclass
class CircuitBreakerSceneCfg(CircuitBreakerSceneBase):
    """Spot + shared HiveBoard circuit breaker scene."""

    robot: ArticulationCfg = SPOT_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(SPOT_EE)
    finger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_FNGR_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_BREAKER_CONTACT_FILTER,
    )
    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_JAW_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_BREAKER_CONTACT_FILTER,
    )

    def __post_init__(self):
        # UUC nests links under Geometry/body/...; body names stay arm_link_wr1.
        self.ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_SOURCE_PRIM
        wr1 = "{ENV_REGEX_NS}/Robot/" + SPOT_ARM_UUC_BODY_PRIM
        for frame in self.ee_frame.target_frames:
            if frame.name == "ee_tcp":
                frame.prim_path = wr1
