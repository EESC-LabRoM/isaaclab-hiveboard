from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaBaseFrameIKAction,
    FrankaIKAbsActionCfg,
)


@configclass
class FrankaGateValveSmallActionsCfg(FrankaIKAbsActionCfg):
    """Track the canonical TCP using the corrected base-frame Jacobian."""

    arm_action = FrankaIKAbsActionCfg().arm_action.replace(class_type=FrankaBaseFrameIKAction)
